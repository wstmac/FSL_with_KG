import logging
import os
import random
import shutil
import time
import warnings
import collections
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import tqdm
from utils import configuration
from numpy import linalg as LA
from scipy.stats import mode
from graph import Graph

import datasets
import models

best_prec1 = -1


def main():
    global args, best_prec1, device
    args = configuration.parser_args()


    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model_dir is None:
        args.model_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.save_path = f'{args.save_path}/{args.model_dir}'
        os.makedirs(args.save_path)
    else:
        args.save_path = f'{args.save_path}/{args.model_dir}'

    log = setup_logger('train_log', args.save_path + '/training.log')
    result_log = setup_logger('result_log', args.save_path + '/result.log', display=True)

    if args.log_info and not args.debug:
        for key, value in sorted(vars(args).items()):
            log.info(str(key) + ': ' + str(value))
            result_log.info(str(key) + ': ' + str(value))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # select GPU device
    device = torch.device(f'cuda:{args.gpu}')

    # load knowledge graph
    knowledge_graph = Graph()
    classFile_to_superclasses, superclassID_to_wikiID =\
        knowledge_graph.class_file_to_superclasses(1, [1,2])

    # create model
    if args.log_info and not args.debug:
        log.info("=> creating model '{}'".format(args.arch))
        result_log.info("=> creating model '{}'".format(args.arch))
    if args.debug:
        import ipdb; ipdb.set_trace()
    model = models.__dict__[args.arch](num_classes=args.num_classes, num_spclasses=len(superclassID_to_wikiID), top_k=args.top_k)

    if args.log_info and not args.debug:
        log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        result_log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = loss_fn()

    optimizer = get_optimizer(model)

    if args.pretrain:
        pretrain = args.pretrain + '/checkpoint.pth.tar'
        if os.path.isfile(pretrain):
            if args.log_info and not args.debug:
                log.info("=> loading pretrained weight '{}'".format(pretrain))
                result_log.info("=> loading pretrained weight '{}'".format(pretrain))
            checkpoint = torch.load(pretrain)
            model_dict = model.state_dict()
            params = checkpoint['state_dict']
            params = {k: v for k, v in params.items() if k in model_dict}
            model_dict.update(params)
            model.load_state_dict(model_dict)
        else:
            if args.log_info and not args.debug:
                log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))
                result_log.info('[Attention]: Do not find pretrained model {}'.format(pretrain))

    # resume from an exist checkpoint
    if os.path.isfile(args.save_path + '/checkpoint.pth.tar') and args.resume == '':
        args.resume = args.save_path + '/checkpoint.pth.tar'

    if args.resume:
        if os.path.isfile(args.resume):
            if args.log_info and not args.debug:
                log.info("=> loading checkpoint '{}'".format(args.resume))
                result_log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            # scheduler.load_state_dict(checkpoint['scheduler'])
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.log_info and not args.debug:
                log.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
                result_log.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(args.resume, checkpoint['epoch']))
        else:
            if args.log_info and not args.debug:
                log.info('[Attention]: Do not find checkpoint {}'.format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    if args.evaluate:
        do_extract_and_evaluate(model, result_log, classFile_to_superclasses)
        return

    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader = get_dataloader('train', not args.disable_train_augment, sample=sample_info, spclasses_dict=classFile_to_superclasses)
    else:
        train_loader = get_dataloader('train', not args.disable_train_augment, shuffle=True, spclasses_dict=classFile_to_superclasses)

    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query] # 5-way 1-shot 15 val quereis
    val_loader = get_dataloader('val', False, sample=sample_info, spclasses_dict=classFile_to_superclasses)

    scheduler = get_scheduler(len(train_loader), optimizer)
    tqdm_loop = warp_tqdm(list(range(args.start_epoch, args.epochs)))
    for epoch in tqdm_loop:
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, scheduler, log)
        # evaluate on meta validation set
        is_best = False
        if (epoch + 1) % args.meta_val_interval == 0:
            prec1 = meta_val(val_loader, model)
            log.info('Meta Val {}: {}'.format(epoch, prec1))
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not args.disable_tqdm:
                tqdm_loop.set_description('Best Acc {:.2f}'.format(best_prec1 * 100.))

        # remember best prec@1 and save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            # 'scheduler': scheduler.state_dict(),
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, folder=args.save_path)

        scheduler.step()

    # do evaluate at the end
    do_extract_and_evaluate(model, result_log, classFile_to_superclasses)


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, sp_gallery, sp_query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)

    sp_gallery = gallery.view(sp_gallery.shape[0], -1)
    sp_query = query.view(sp_query.shape[0], -1)

    distance = get_metric(metric_type)(gallery, query)
    sp_distance = get_metric(metric_type)(sp_gallery, sp_query)

    combo_distance = args.loss_alpha * distance + args.loss_beta * sp_distance

    predict = torch.argmin(combo_distance, dim=1)
    predict = torch.take(train_label, predict)

    return predict


def meta_val(test_loader, model, train_mean=None):
    top1 = AverageMeter()
    model.eval()
    if args.debug:
        import ipdb; ipdb.set_trace()
    with torch.no_grad():
        tqdm_test_loader = warp_tqdm(test_loader)
        for i, (inputs, target, sp_target) in enumerate(tqdm_test_loader):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            sp_target = sp_target.to(device, non_blocking=True)
            output, _, sp_output, _ = model(inputs)
            if train_mean is not None:
                output = output - train_mean

            # get image feature
            train_out = output[:args.meta_val_way * args.meta_val_shot]
            train_label = target[:args.meta_val_way * args.meta_val_shot]

            test_out = output[args.meta_val_way * args.meta_val_shot:]
            test_label = target[args.meta_val_way * args.meta_val_shot:]

            train_out = train_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1)
            train_label = train_label[::args.meta_val_shot]

            # get super class feature
            train_sp_out = sp_output[:args.meta_val_way * args.meta_val_shot]
            test_sp_out = sp_output[args.meta_val_way * args.meta_val_shot:]
            train_sp_out = train_sp_out.reshape(args.meta_val_way, args.meta_val_shot, -1).mean(1)

            prediction = metric_prediction(train_out, test_out, train_sp_out, test_sp_out, train_label, args.meta_val_metric)
            acc = (prediction == test_label).float().mean()
            top1.update(acc.item())
            if not args.disable_tqdm:
                tqdm_test_loader.set_description('Meta Val Acc {:.2f}'.format(top1.avg * 100))
    return top1.avg


def train(train_loader, model, criterion, optimizer, epoch, scheduler, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    tqdm_train_loader = warp_tqdm(train_loader)
    for i, (input, target, sp_target) in enumerate(tqdm_train_loader):
        if args.scheduler == 'cosine':
            scheduler.step(epoch * len(train_loader) + i)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.do_meta_train:
            target = torch.arange(args.meta_train_way)[:, None].repeat(1, args.meta_train_query).reshape(-1).long()
        target = target.to(device, non_blocking=True)
        sp_target = sp_target.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)
        # compute output
        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).to(device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output, _, _, _ = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            _, output, _, sp_output = model(input)
            if args.do_meta_train:
                # output = output.to(device)
                shot_proto = output[:args.meta_train_shot * args.meta_train_way]
                query_proto = output[args.meta_train_shot * args.meta_train_way:]
                shot_proto = shot_proto.reshape(args.meta_train_way, args.meta_train_shot, -1).mean(1)
                output = -get_metric(args.meta_train_metric)(shot_proto, query_proto)
            loss = criterion(output, target, sp_output, sp_target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        if not args.disable_tqdm:
            tqdm_train_loader.set_description(f'Epoch {epoch:2d} Top1 Acc: {top1.avg:.2f}')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log.info('Epoch: [{:2d}][{:3d}/{:3d}]\t'
                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                     'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


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


def get_scheduler(batches, optimiter):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """
    SCHEDULER = {'step': StepLR(optimiter, args.lr_stepsize, args.lr_gamma),
                 'multi_step': MultiStepLR(optimiter, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)],
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimiter, batches * args.epochs, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]


def get_optimizer(module):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def extract_feature(train_loader, val_loader, model, tag='last'):
    # return out mean, fcout mean, out feature, fcout features
    save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
    if os.path.isfile(save_dir + '/output.plk'):
        data = load_pickle(save_dir + '/output.plk')
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, sp_out_mean, fc_out_mean = [], [], []
        for i, (inputs, _, _) in enumerate(warp_tqdm(train_loader)):
            inputs = inputs.to(device)
            outputs, fc_outputs, sp_outputs, _ = model(inputs)
            out_mean.append(outputs.cpu().data.numpy())
            sp_out_mean.append(sp_outputs.cpu().data.numpy())
            if fc_outputs is not None:
                fc_out_mean.append(fc_outputs.cpu().data.numpy())
        out_mean = np.concatenate(out_mean, axis=0).mean(0)
        sp_out_mean = np.concatenate(sp_out_mean, axis=0).mean(0)
        if len(fc_out_mean) > 0:
            fc_out_mean = np.concatenate(fc_out_mean, axis=0).mean(0)
        else:
            fc_out_mean = -1

        output_dict = collections.defaultdict(list)
        fc_output_dict = collections.defaultdict(list)
        for i, (inputs, labels, _) in enumerate(warp_tqdm(val_loader)):
            # compute output
            inputs = inputs.to(device)
            outputs, fc_outputs, sp_outputs, _ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            sp_outputs = sp_outputs.cpu().data.numpy()
            if fc_outputs is not None:
                fc_outputs = fc_outputs.cpu().data.numpy()
            else:
                fc_outputs = [None] * outputs.shape[0]
            for out, sp_out, fc_out, label in zip(outputs, sp_outputs, fc_outputs, labels):
                output_dict[label.item()].append([out, sp_out])
                fc_output_dict[label.item()].append(fc_out)
        # output_dict = np.asarray(output_dict)
        all_info = [out_mean, sp_out_mean, fc_out_mean, output_dict, fc_output_dict]
        save_pickle(save_dir + '/output.plk', all_info)
        return all_info


def get_dataloader(split, aug=False, shuffle=True, out_name=False, sample=None, spclasses_dict=None):
    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(84, disable_random_resize=args.disable_random_resize)
    else:
        transform = datasets.without_augment(84, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, transform, out_name=out_name, spclasses_dict=spclasses_dict)

    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.workers, pin_memory=True)
    return loader


def warp_tqdm(data_loader):
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def load_checkpoint(model, type='best'):
    if type == 'best':
        checkpoint = torch.load('{}/model_best.pth.tar'.format(args.save_path))
    elif type == 'last':
        checkpoint = torch.load('{}/checkpoint.pth.tar'.format(args.save_path))
    else:
        assert False, 'type should be in [best, or last], but got {}'.format(type)
    model.load_state_dict(checkpoint['state_dict'])


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def meta_evaluate(data, train_mean, sp_train_mean, shot):
    un_list = []
    l2n_list = []
    cl2n_list = []
    for _ in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, sp_train_data, sp_test_data, train_label, test_label = sample_case(data, shot)
        acc = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
                                train_mean=train_mean, sp_train_mean=sp_train_mean, norm_type='CL2N')
        cl2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
                                train_mean=train_mean, sp_train_mean=sp_train_mean, norm_type='L2N')
        l2n_list.append(acc)
        acc = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
                                train_mean=train_mean, sp_train_mean=sp_train_mean, norm_type='UN')
        un_list.append(acc)
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    return un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf


def metric_class_type(gallery, query, sp_gallery, sp_query, train_label, test_label, shot, train_mean=None, sp_train_mean=None, norm_type='CL2N'):
    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]

        sp_gallery = sp_gallery - sp_train_mean
        sp_gallery = sp_gallery / LA.norm(sp_gallery, 2, 1)[:, None]
        sp_query = sp_query - sp_train_mean
        sp_query = sp_query / LA.norm(sp_query, 2, 1)[:, None]


    elif norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

        sp_gallery = sp_gallery / LA.norm(sp_gallery, 2, 1)[:, None]
        sp_query = sp_query / LA.norm(sp_query, 2, 1)[:, None]

    gallery = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)
    sp_gallery = sp_gallery.reshape(args.meta_val_way, shot, sp_gallery.shape[-1]).mean(1)
    train_label = train_label[::shot]
    subtract = gallery[:, None, :] - query
    sp_subtract = sp_gallery[:, None, :] - sp_query


    if args.debug:
        import ipdb; ipdb.set_trace()
    distance = LA.norm(subtract, 2, axis=-1)
    sp_distance = LA.norm(sp_subtract, 2, axis=-1)

    combo_distance = args.loss_alpha * distance + args.loss_beta * sp_distance

    idx = np.argpartition(combo_distance, args.num_NN, axis=0)[:args.num_NN]
    nearest_samples = np.take(train_label, idx)
    out = mode(nearest_samples, axis=0)[0]
    out = out.astype(int)
    test_label = np.array(test_label)
    acc = (out == test_label).mean()
    return acc


def sample_case(ld_dict, shot):
    sample_class = random.sample(list(ld_dict.keys()), args.meta_val_way)
    train_input = []
    test_input = []
    test_label = []
    train_label = []
    sp_train_input = []
    sp_test_input = []
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + args.meta_val_query)
        # if args.debug:
        #     import ipdb; ipdb.set_trace()
        train_label += [each_class] * len(samples[:shot])
        test_label += [each_class] * len(samples[shot:])
        train_input += [sample[0] for sample in samples[:shot]]
        test_input += [sample[0] for sample in samples[shot:]]

        sp_train_input += [sample[1] for sample in samples[:shot]]
        sp_test_input += [sample[1] for sample in samples[shot:]]

    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    sp_train_input = np.array(sp_train_input).astype(np.float32)
    sp_test_input = np.array(sp_test_input).astype(np.float32)
    return train_input, test_input, sp_train_input, sp_test_input, train_label, test_label
    # train_data, test_data, sp_train_data, sp_test_data, train_label, test_label


def do_extract_and_evaluate(model, log, classFile_to_superclasses):
    train_loader = get_dataloader('train', aug=False, shuffle=False, out_name=False, spclasses_dict=classFile_to_superclasses)
    val_loader = get_dataloader('test', aug=False, shuffle=False, out_name=False, spclasses_dict=classFile_to_superclasses)
    load_checkpoint(model, 'last')
    out_mean, sp_out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'last')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, sp_out_mean, 1)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, sp_out_mean, 5)
    log.info(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    if args.eval_fc:
        accuracy_info = meta_evaluate(fc_out_dict, fc_out_mean, 1)
        print('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format('Logits', *accuracy_info))
        log.info('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format('Logits', *accuracy_info))
    load_checkpoint(model, 'best')
    out_mean, sp_out_mean, fc_out_mean, out_dict, fc_out_dict = extract_feature(train_loader, val_loader, model, 'best')
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, sp_out_mean, 1)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, sp_out_mean, 5)
    log.info(
        'Meta Test: BEST\nfeature\tUN\tL2N\tCL2N\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
            'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    if args.eval_fc:
        accuracy_info = meta_evaluate(fc_out_dict, fc_out_mean, 1)
        print('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format('Logits', *accuracy_info))
        log.info('{}\t{:.4f}\t{:.4f}\t{:.4f}'.format('Logits', *accuracy_info))


def loss_fn():

    def _loss_fn(class_outputs, labels, sp_outputs, sp_labels):
        BCE_loss = F.binary_cross_entropy_with_logits(sp_outputs, sp_labels)
        CEL_loss = F.cross_entropy(class_outputs, labels)
        
        combo_loss = CEL_loss * args.loss_alpha + BCE_loss * args.loss_beta

        return combo_loss

    return _loss_fn

if __name__ == '__main__':
    main()