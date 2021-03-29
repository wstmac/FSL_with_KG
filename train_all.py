import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
import os
from sentence_transformers import SentenceTransformer

from datasets import MiniImageNet, SupportingSetSampler, prepare_nshot_task
import models
from utils import compute_confidence_interval, get_splits, AverageMeter, setup_logger, get_classFile_to_wikiID, argmax_evaluation
from graph import Graph, extract_embedding_by_labels, find_nodeIndex_by_imgLabels


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model_arch', default='conv4', choices=['conv4', 'resnet10', 'resnet18'], type=str)
    # parser.add_argument('--attention', action='store_true')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--num_epoch', default=90, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--scheduler_milestones', nargs='+', type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--model_saving_rate', default=30, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--support_groups', default=1000, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluation_rate', default=10, type=int)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save_settings', action='store_true')
    parser.add_argument('--layer', default=4, type=int)
    # parser.add_argument('--gcn_path', type=str)
    # parser.add_argument('--img_encoder_path', type=str)


    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    model_arch = args.model_arch
    # attention = args.attention
    learning_rate = args.learning_rate
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    start_epoch = args.start_epoch
    num_epoch = args.num_epoch
    model_saving_rate = args.model_saving_rate
    toTrain = args.train
    toEvaluate = args.evaluate
    evaluation_rate = args.evaluation_rate
    checkpoint = args.checkpoint
    normalize = args.normalize
    scheduler_milestones = args.scheduler_milestones
    save_settings = args.save_settings
    support_groups = args.support_groups

    # gcn_path = args.gcn_path
    # img_encoder_path = args.img_encoder_path

    # ------------------------------- #
    # Generate folder 
    # ------------------------------- #
    if checkpoint:
        model_dir = f'./training_models/{args.model_dir}'
    else:
        model_dir = f'./training_models/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(model_dir)


    
    # ------------------------------- #
    # Config logger
    # ------------------------------- #
    train_logger = setup_logger('train_logger', f'{model_dir}/train_all.log')
    result_logger = setup_logger('result_logger', f'{model_dir}/result_all.log')
    if save_settings:
        # ------------------------------- #
        # Saving training parameters
        # ------------------------------- #
        result_logger.info(f'Model: {model_arch}')
        result_logger.info(f'Attention Layer: {args.layer}')
        result_logger.info(f'Learning rate: {learning_rate}')
        result_logger.info(f'Alpha:{alpha} Beta:{beta} Gamma:{gamma}')
        # result_logger.info(f'alpha: {alpha}')
        result_logger.info(f'Normalize feature vector: {normalize}')

    # ------------------------------- #
    # Load extracted knowledge graph
    # ------------------------------- #
    knowledge_graph = Graph()
    classFile_to_superclasses, superclassID_to_wikiID =\
        knowledge_graph.class_file_to_superclasses(1, [1,2])
    nodes = knowledge_graph.nodes
    # import ipdb; ipdb.set_trace()

    layer = 2
    layer_nums = [768, 2048, 1600]
    edges = knowledge_graph.edges

    cat_feature = 3200
    final_feature = 1024

    ####################
    # Prepare Data Set #
    ####################
    print('preparing dataset')
    base_cls, val_cls, support_cls = get_splits()

    base = MiniImageNet('base', base_cls, val_cls, support_cls, classFile_to_superclasses)
    base_loader = DataLoader(base, batch_size=256, shuffle=True, num_workers=4)


    support = MiniImageNet('support', base_cls, val_cls, support_cls,
            classFile_to_superclasses, eval=True)
    support_loader_1 = DataLoader(support,
                    batch_sampler=SupportingSetSampler(support, 1, 5, 15, support_groups),
                    num_workers=4)
    support_loader_5 = DataLoader(support,
                    batch_sampler=SupportingSetSampler(support, 5, 5, 15, support_groups),
                    num_workers=4)


    #########
    # Model #
    #########
    # sentence transformer
    sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')

    # image encoder
    if model_arch == 'conv4':
        img_encoder = models.Conv4Attension(len(base_cls), len(superclassID_to_wikiID))

    if model_arch == 'resnet10':
        img_encoder = models.resnet10(len(base_cls), len(superclassID_to_wikiID))

    if model_arch == 'resnet18':
        img_encoder = models.resnet18(len(base_cls), len(superclassID_to_wikiID))

    # img_encoder.load_state_dict(torch.load(f'{model_dir}/{img_encoder_path}'))
    # img_encoder.to(device)
    # img_encoder.eval()


    # knowledge graph encoder
    GCN = models.GCN(layer, layer_nums, edges)
    # GCN.load_state_dict(torch.load(f'{model_dir}/{gcn_path}'))
    # GCN.to(device)
    # GCN.eval()

    # total model
    model = models.FSKG(cat_feature, final_feature, img_encoder, GCN, len(base_cls))
    model.to(device)
    

    # loss function and optimizer
    criterion = loss_fn(alpha, beta, gamma, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=0.1)

    if save_settings:
        result_logger.info('optimizer: torch.optim.SGD(model.parameters(), '
            f'lr={learning_rate}, momentum=0.9, weight_decay=1e-4, nesterov=True)')
        result_logger.info(f'scheduler: MultiStepLR(optimizer, milestones={scheduler_milestones}, gamma=0.1)\n')
        # result_logger.info('='*40+'Results Below'+'='*40+'\n')

    if checkpoint:
        print('load model...')
        model.load_state_dict(torch.load(f'{model_dir}/FSKG_{start_epoch-1}.pth'))
        model.to(device)

        # for _ in range(start_epoch - 1):
        #     scheduler.step()

    # ---------------------------------------- #
    # Graph convolution to get kg embeddings
    # ---------------------------------------- #

    # encode node description
    desc_embeddings = knowledge_graph.encode_desc(sentence_transformer).to(device)

    # start graph convolution
    # import ipdb; ipdb.set_trace()
    # kg_embeddings = GCN(desc_embeddings)
    # kg_embeddings = kg_embeddings.to('cpu')


    classFile_to_wikiID = get_classFile_to_wikiID()
    # train_class_name_to_id = base.class_name_to_id
    train_id_to_class_name = base.id_to_class_name
    # eval_class_name_to_id = support.class_name_to_id
    eval_id_to_class_name = support.id_to_class_name

    # ------------------------------- #
    # Start to train
    # ------------------------------- #
    if toTrain:
        for epoch in range(start_epoch, start_epoch+num_epoch):
            model.train()
            train(model, img_encoder, normalize, base_loader, optimizer, criterion, epoch,
                    start_epoch+num_epoch-1, device, train_logger, 
                    nodes, desc_embeddings, train_id_to_class_name, classFile_to_wikiID)
            scheduler.step()

            if epoch % model_saving_rate == 0:
                torch.save(model.state_dict(), f'{model_dir}/FSKG_{epoch}.pth')

                # ------------------------------- #
                # Evaluate current model
                # ------------------------------- #
            if toEvaluate:
                if epoch % evaluation_rate == 0:
                    evaluate(model, normalize, epoch, support_loader_1,
                            1, 5, 15, device, result_logger, nodes, desc_embeddings, eval_id_to_class_name, classFile_to_wikiID)
                    evaluate(model, normalize, epoch, support_loader_5,
                            5, 5, 15, device, result_logger, nodes, desc_embeddings, eval_id_to_class_name, classFile_to_wikiID)

    else:
        # pass
        if toEvaluate:
            evaluate(model, normalize, 30, support_loader_1,
                    1, 5, 15, device, result_logger, nodes, desc_embeddings, eval_id_to_class_name, classFile_to_wikiID)
            evaluate(model, normalize, 30, support_loader_5,
                    5, 5, 15, device, result_logger, nodes, desc_embeddings, eval_id_to_class_name, classFile_to_wikiID)
    result_logger.info('='*140)


def train(model, img_encoder, normalize, base_loader, optimizer, criterion, epoch,
            total_epoch, device, logger, nodes, desc_embeddings, id_to_class_name, classFile_to_wikiID):
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    model.train()
    img_encoder.eval()
    start = time.time()

    for i, (imgs, labels, sp_labels) in enumerate(base_loader):
        data_time.update(time.time() - start)
        imgs = imgs.to(device)
        labels = labels.to(device)
        sp_labels = sp_labels.to(device)

        corr_nodeIndexs = find_nodeIndex_by_imgLabels(nodes, labels, id_to_class_name, classFile_to_wikiID) 

        _, class_outputs, sp_outputs, att_features, corr_features = model(imgs, desc_embeddings, corr_nodeIndexs, norm=normalize)
        loss = criterion(class_outputs, sp_outputs, labels, sp_labels, att_features, corr_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        if i % 30 == 29:    # print every 30 mini-batches
            logger.info(f'[{epoch:3d}/{total_epoch}|{i+1:3d}, '
                f'{len(base_loader)}] batch_time: {batch_time.avg:.2f} '
                f'data_time: {data_time.avg:.2f} loss: {losses.avg:.3f}')  

            batch_time.reset() 
            data_time.reset() 
            losses.reset() 


def evaluate(model, normalize, epoch, support_loader, n, k, q, device, logger, nodes, desc_embeddings, id_to_class_name, classFile_to_wikiID):
    accs_l2 = []
    accs_cosine = []
    model.eval()

    with torch.no_grad():
        for data in tqdm(support_loader):
            imgs, labels  = prepare_nshot_task(n, k, q, data, device)
            support_corr_nodeIndexs = find_nodeIndex_by_imgLabels(nodes, data[1][:n*k], id_to_class_name, classFile_to_wikiID) 
            support_imgs, _, _, _, _ = model(imgs[:n*k], desc_embeddings, support_corr_nodeIndexs, norm=normalize)

            queries = []
            for i in range(k):
                query_corr_nodeIndexs = find_nodeIndex_by_imgLabels(nodes, (q*k)*[data[1][0+i*n]], id_to_class_name, classFile_to_wikiID) 
                query_imgs, _, _, _, _ = model(imgs[n*k:], desc_embeddings, query_corr_nodeIndexs, norm=normalize)
                queries.append(query_imgs)
            # import ipdb; ipdb.set_trace()

            acc_l2 = argmax_evaluation(support_imgs, queries, labels, n, k, q, 'l2')
            acc_cosine = argmax_evaluation(support_imgs, queries, labels, n, k, q, 'cosine')
            accs_l2.append(acc_l2)
            accs_cosine.append(acc_cosine)
    m_l2, pm_l2 = compute_confidence_interval(accs_l2)
    m_cosine, pm_cosine = compute_confidence_interval(accs_cosine)
    logger.info(f'{epoch:3d}.pth: {n}-shot \t l2: {m_l2:.2f}+/-{pm_l2:.2f} \t '
            f'cosine: {m_cosine:.2f}+/-{pm_cosine:.2f}')



def loss_fn(alpha, beta, gamma, device):

    def _loss_fn(class_outputs, sp_outputs, labels, sp_labels, att_features, corr_features):
        # import ipdb; ipdb.set_trace()
        loss_target = torch.ones(att_features.shape[0]).to(device)
        BCE_loss = F.binary_cross_entropy_with_logits(sp_outputs, sp_labels)
        CEL_loss = F.cross_entropy(class_outputs, labels)
        Feature_loss = F.cosine_embedding_loss(att_features, corr_features, loss_target)
        
        combo_loss = CEL_loss * alpha + BCE_loss * beta + Feature_loss * gamma

        return combo_loss

    return _loss_fn


if __name__ == '__main__':
    main()
