# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, gpu, dim=128, K=2048, m=0.999, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.temperature = T
        self.base_temperature = 0.07
        self.gpu = gpu

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=64, sp_embedding_feature_dim = 1024, pool_type='avg_pool', top_k=32)
        self.encoder_k = base_encoder(num_classes=64, sp_embedding_feature_dim = 1024, pool_type='avg_pool', top_k=32)


        # dim_mlp = self.encoder_q.fc.weight.shape[1]
        # self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        # self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_labels", torch.zeros(K, dtype=torch.long)-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, contrastive_k, batch_labels):
        # gather keys before updating queue
        contrastive_k = concat_all_gather(contrastive_k)
        batch_labels = concat_all_gather(batch_labels)

        batch_size = contrastive_k.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = contrastive_k.T
        self.queue_labels[ptr:ptr + batch_size] = batch_labels

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, batch_labels):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        batch_size = im_q.shape[0]

        # compute query features
        _, output, _, contrastive_q = self.encoder_q(im_q)  # queries: NxC
        contrastive_q = nn.functional.normalize(contrastive_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            _, _, _, contrastive_k = self.encoder_k(im_k)  # keys: NxC
            contrastive_k = nn.functional.normalize(contrastive_k, dim=1)

            # undo shuffle
            contrastive_k = self._batch_unshuffle_ddp(contrastive_k, idx_unshuffle)


        self._dequeue_and_enqueue(contrastive_k, batch_labels)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrastive_q, self.queue.clone().detach()),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # ----------------------------------------------------------- #
        # mask: use class as mask
        # ----------------------------------------------------------- #
        mask = torch.eq(batch_labels.reshape(-1,1).repeat(1,self.K), self.queue_labels).float().cuda(self.gpu)

        # mask-out self-contrast cases
        # logits_mask = torch.scatter(
        #     torch.ones_like(mask),
        #     1,
        #     torch.arange(batch_size).view(-1, 1).cuda(self.gpu),
        #     0
        # )
        logits_mask = torch.ones_like(mask)
        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / torch.where(mask.sum(1)==0, torch.tensor(1, dtype=torch.float, device=f'cuda:{self.gpu}'), mask.sum(1))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # sp_loss
        sp_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        sp_loss = sp_loss.view(1, batch_size).mean()



        return output, sp_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ComboLoss(nn.Module):
    def __init__(self, loss_alpha, loss_beta):
        super(ComboLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta

        self.CEL = nn.CrossEntropyLoss()
        # self.supConLoss = SupConLoss()

    def forward(self, class_outputs, labels, sp_loss):
        c_loss = self.CEL(class_outputs, labels)
        # sp_loss = self.supConLoss(sp_outputs, sp_mask)
        combo_loss = c_loss * self.loss_alpha + sp_loss * self.loss_beta

        return c_loss, combo_loss