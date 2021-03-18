import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
import os

from datasets import MiniImageNet, SupportingSetSampler, prepare_nshot_task
import models
from utils import compute_confidence_interval, get_splits, evaluation, AverageMeter, setup_logger
from graph import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model_arch', default='conv4', choices=['conv4', 'resnet10', 'resnet50'], type=str)
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--num_epoch', default=90, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--scheduler_milestones', nargs='+', type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--model_saving_rate', default=30, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--support_groups', default=10000, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--save_settings', action='store_true')



    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    model_arch = args.model_arch
    attention = args.attention
    learning_rate = args.learning_rate
    alpha = args.alpha
    start_epoch = args.start_epoch
    num_epoch = args.num_epoch
    model_saving_rate = args.model_saving_rate
    toTrain = args.train
    toEvaluate = args.evaluate
    checkpoint = args.checkpoint
    normalize = args.normalize
    scheduler_milestones = args.scheduler_milestones
    save_settings = args.save_settings
    support_groups = args.support_groups

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
    train_logger = setup_logger('train_logger', f'{model_dir}/train.log')
    result_logger = setup_logger('result_logger', f'{model_dir}/result.log')
    if save_settings:
        # ------------------------------- #
        # Saving training parameters
        # ------------------------------- #
        result_logger.info(f'Model: {model_arch}\tAttention: {attention}')
        result_logger.info(f'Learning rate: {learning_rate}')
        result_logger.info(f'alpha: {alpha}')
        result_logger.info(f'Normalize feature vector: {normalize}')
    # ------------------------------- #
    # Load extracted knowledge graph
    # ------------------------------- #
    knowledge_graph = Graph()
    classFile_to_superclasses, superclassID_to_wikiID =\
        knowledge_graph.class_file_to_superclasses(1, [1,2])


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
    if model_arch == 'conv4':
        if attention:
            model = models.Conv4Attension(len(base_cls), len(superclassID_to_wikiID))
        else:
            model = models.Conv4Classifier(len(base_cls))

    if model_arch == 'resnet10':
        model = models.resnet10(attention, len(base_cls), len(superclassID_to_wikiID))

    if model_arch == 'resnet50':
        model = models.resnet50(attention, len(base_cls), len(superclassID_to_wikiID))

    model.to(device)

    # loss function and optimizer
    criterion = loss_fn(alpha)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=scheduler_milestones, gamma=0.1)

    if save_settings:
        result_logger.info('optimizer: torch.optim.SGD(model.parameters(), '
            f'lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)')
        result_logger.info(f'scheduler: MultiStepLR(optimizer, milestones={scheduler_milestones}, gamma=0.1)\n')
        result_logger.info('='*40+'Results Below'+'='*40+'\n')

    if checkpoint:
        print('load model...')
        model.load_state_dict(torch.load(f'{model_dir}/{start_epoch-1}.pth'))
        model.to(device)

        for _ in range(start_epoch - 1):
            scheduler.step()



    # ------------------------------- #
    # Start to train
    # ------------------------------- #
    if toTrain:
        for epoch in range(start_epoch, start_epoch+num_epoch):
            model.train()
            train(model, normalize, base_loader, optimizer, criterion, epoch,
                    start_epoch+num_epoch-1, device, train_logger)
            scheduler.step()

            if epoch % model_saving_rate == 0:
                torch.save(model.state_dict(), f'{model_dir}/{epoch}.pth')

                # ------------------------------- #
                # Evaluate current model
                # ------------------------------- #
                if toEvaluate:
                    evaluate(model, normalize, epoch, support_loader_1,
                            1, 5, 15, result_logger)
                    evaluate(model, normalize, epoch, support_loader_5,
                            5, 5, 15, result_logger)
    else:
        if toEvaluate:
            evaluate(model, normalize, start_epoch-1, support_loader_1,
                    1, 5, 15, result_logger)
            evaluate(model, normalize, start_epoch-1, support_loader_5,
                    5, 5, 15, result_logger)


def train(model, normalize, base_loader, optimizer, criterion, epoch,
            total_epoch, device, logger):
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    model.train()
    start = time.time()

    for i, (imgs, labels, sp_labels) in enumerate(base_loader):
        data_time.update(time.time() - start)

        imgs = imgs.to(device)
        labels = labels.to(device)
        sp_labels = sp_labels.to(device)

        _, class_outputs, sp_outputs = model(imgs, norm=normalize)
        loss = criterion(class_outputs, sp_outputs, labels, sp_labels)

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


def evaluate(model, normalize, epoch, support_loader, n, k, q, logger):
    accs_l2 = []
    accs_cosine = []
    model.eval()

    with torch.no_grad():
        for data in tqdm(support_loader):
            imgs, labels  = prepare_nshot_task(n, k, q, data)
            _, outputs, _ = model(imgs, norm=normalize)

            acc_l2 = evaluation(outputs, labels, n, k, q, 'l2')
            acc_cosine = evaluation(outputs, labels, n, k, q, 'cosine')
            accs_l2.append(acc_l2)
            accs_cosine.append(acc_cosine)
    
    m_l2, pm_l2 = compute_confidence_interval(accs_l2)
    m_cosine, pm_cosine = compute_confidence_interval(accs_cosine)
    # file_writer.write(f'{epoch:3d}.pth {n}-shot\tAccuracy_l2: {m_l2:.2f}+/-{pm_l2:.2f} Accuracy_cosine: {m_cosine:.2f}+/-{pm_cosine:.2f}\n')
    logger.info(f'{epoch:3d}.pth: {n}-shot \t l2: {m_l2:.2f}+/-{pm_l2:.2f} \t '
                f'cosine: {m_cosine:.2f}+/-{pm_cosine:.2f}')


def loss_fn(alpha):

    def _loss_fn(class_outputs, sp_outputs, labels, sp_labels):
        # import ipdb; ipdb.set_trace()
        BCE_loss = F.binary_cross_entropy_with_logits(sp_outputs, sp_labels)
        CEL_loss = F.cross_entropy(class_outputs, labels)
        
        combo_loss = CEL_loss * alpha + BCE_loss * (1 - alpha)

        return combo_loss

    return _loss_fn


if __name__ == '__main__':
    main()
