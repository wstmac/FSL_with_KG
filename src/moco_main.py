import logging
import os
import random
import shutil
import time
import math
import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import datasets
import models
from graph import Graph
from utils import configuration, miscellaneous
from moco.builder import MoCo


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    args = configuration.parser_args()

    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model_dir is None:
        args.model_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.save_path = f'{args.save_path}/{args.model_dir}'
        os.makedirs(args.save_path)
    else:
        args.save_path = f'{args.save_path}/{args.model_dir}'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # load sentence transformer
    sentence_transformer = SentenceTransformer('stsb-roberta-large')

    # load knowledge graph
    knowledge_graph = Graph()
    classFile_to_superclasses, _ =\
        knowledge_graph.class_file_to_superclasses(1, [1,2], sentence_transformer, 1)

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, classFile_to_superclasses))




def main_worker(gpu, ngpus_per_node, args, classFile_to_superclasses):

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    args.rank = gpu
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001',
                            world_size=args.world_size, rank=args.rank)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)


    log = setup_logger('train', f'{args.save_path}/train_{args.rank}.log')

    # Data loading code
    used_files = set(miscellaneous.get_classFile_to_wikiID().keys())
    train_sampler, train_loader = get_dataloader('train', used_files, not args.disable_train_augment, shuffle=False, spclasses_dict=classFile_to_superclasses, args=args)


    # ----------------------------------------------------------- #
    # Build model
    # ----------------------------------------------------------- #
    model = MoCo(models.__dict__[args.arch], args.moco_dim, args.moco_k, args.moco_m, args.moco_t)

    optimizer = get_optimizer(model, args)

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)


    model = DDP(model, device_ids=[args.gpu])


    # resume from an exist checkpoint
    if os.path.isfile(args.save_path + '/checkpoint.pth.tar') and args.resume == '':
        args.resume = args.save_path + '/checkpoint.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])


    cudnn.benchmark = True


    # scheduler = get_scheduler(len(train_loader), optimizer, args)
    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)), args)


    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    for epoch in tqdm_loop:
        train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log)

        # remember best prec@1 and save checkpoint
        if args.rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=False, folder=args.save_path)



def train(train_loader, model, criterion, optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cel_losses = AverageMeter()
    sp_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # tqdm_train_loader = warp_tqdm(train_loader, args)
    for i, (input, target, _) in enumerate(train_loader):

        # print(f'In {args.gpu}, it process {i}th data loader')

        # measure data loading time
        data_time.update(time.time() - end)

        input_q = input[0].cuda(args.gpu, non_blocking=True)
        input_k = input[1].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # print(f'In {args.gpu}, before pass to model')


        output, target = model(input_q, input_k)

        # ----------------------------------------------------------- #
        # Contrative loss version
        # ----------------------------------------------------------- #
        # print(f'In {args.gpu}, after pass to model')
        loss = criterion(output, target)
        # print(f'In {args.gpu}, the {i}th loss is {loss.item()}')

        # measure accuracy and record loss
        losses.update(loss.item(), input_q.size(0))
        # cel_losses.update(cel_loss.item(), input_q.size(0))
        # sp_losses.update(sp_loss.item(), input_q.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input_q.size(0))
        top5.update(prec5[0], input_q.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            # if args.gpu == 0:
            #     print('Epoch: [{:2d}][{:3d}/{:3d}]\t'
            #             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #         epoch, i, len(train_loader), batch_time=batch_time,
            #         data_time=data_time, loss=losses, top1=top1, top5=top5))
            log.info('Epoch: [{:2d}][{:3d}/{:3d}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
                # print('Epoch: [{:2d}][{:3d}/{:3d}]\t'
                #         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                #         'CEL Loss {cel_loss.val:.4f} ({cel_loss.avg:.4f})\t'
                #         'SP Loss {sp_loss.val:.4f} ({sp_loss.avg:.4f})\t'
                #         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                #         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                #     epoch, i, len(train_loader), batch_time=batch_time,
                #     data_time=data_time, cel_loss=cel_losses, sp_loss=sp_losses,
                #     loss=losses, top1=top1, top5=top5))




def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', folder='result/default'):
    torch.save(state, folder + '/' + filename)
    if is_best:
        shutil.copyfile(folder + '/' + filename, folder + '/model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def setup_logger(name, log_file, level=logging.INFO, display=False):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter(
        "[%(asctime)s %(filename)s:%(lineno)3s] %(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    if display:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger



def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # if args.cos:  # cosine lr schedule
    #     lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    # else:  # stepwise lr schedule
    #     for milestone in args.schedule:
    #         lr *= 0.1 if epoch >= milestone else 1.
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(module, args):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]


def warp_tqdm(data_loader, args):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def get_dataloader(split, used_files, aug=False, shuffle=True, out_name=False, sample=None, spclasses_dict=None, args=None):
    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(84, disable_random_resize=args.disable_random_resize)
        transform = datasets.TwoCropTransform(transform)
    else:
        transform = datasets.without_augment(84, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, used_files, transform, out_name=out_name, spclasses_dict=spclasses_dict)

    sampler = torch.utils.data.distributed.DistributedSampler(sets)
    loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                         num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)
    # print(f'args.batch_size: {args.batch_size}\targs.workers:{args.workers}')
    return sampler, loader


if __name__ == '__main__':
    main()
