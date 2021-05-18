import torch
import torch.nn as nn

class SupMocoLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, gpu, temperature=0.07, 
                 base_temperature=0.07, 
                 dim=128, K=192):
        """
        dim: feature dimension after the mlp (default: 128)
        K: queue size (default: 1920) store both query and key
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SupMocoLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.gpu = gpu

        # ----------------------------------------------------------- #
        # Supervised Moco
        # ----------------------------------------------------------- #
        self.K = K
        
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_labels", torch.zeros(K, dtype=torch.long)-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, contrastive_q, contrastive_k, batch_labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            contrastive_q: hidden vector of query [bsz, dim]
            contrastive_k: hidden vector of key [bsz, dim]
            batch_labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = contrastive_q.shape[0]

        # dequeue and enqueue
        self._dequeue_and_enqueue(contrastive_k, batch_labels)

        # print(f'In {self.gpu}, after enqueue key')

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
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(self.gpu),
            0
        )
        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.where(mask.sum(1)==0, torch.tensor(1, dtype=torch.float, device=f'cuda:{self.gpu}'), mask.sum(1))

        # sp_loss
        sp_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        sp_loss = sp_loss.view(1, batch_size).mean()

        return sp_loss


    @torch.no_grad()
    def _dequeue_and_enqueue(self, contrastive_k, batch_labels):
        # gather keys before updating queue
        contrastive_k = concat_all_gather(contrastive_k)
        batch_labels = concat_all_gather(batch_labels)

        batch_size = contrastive_k.shape[0]

        ptr = int(self.queue_ptr)

        # print(f'batch size: {batch_size}')
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = contrastive_k.T
        # replace the labels at ptr (dequeue and enqueue)
        self.queue_labels[ptr:ptr + batch_size] = batch_labels
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


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

    def __init__(self, loss_alpha, loss_beta, gpu):
        super(ComboLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta

        self.CEL = nn.CrossEntropyLoss()
        self.SupMocoLoss = SupMocoLoss(gpu)

    def forward(self, class_outputs, labels, contrastive_q, contrastive_k):
        c_loss = self.CEL(class_outputs, labels)
        sp_loss = self.SupMocoLoss(contrastive_q, contrastive_k, labels)
        combo_loss = c_loss * self.loss_alpha + sp_loss * self.loss_beta

        return c_loss, sp_loss, combo_loss
