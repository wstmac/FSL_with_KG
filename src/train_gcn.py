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
from numpy.core.shape_base import block
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
from utils import configuration, miscellaneous
from numpy import linalg as LA
from scipy.stats import mode
from graph import Graph
from sentence_transformers import SentenceTransformer

import datasets
import models
from models import GraphConv

best_prec1 = -1

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    global args, best_prec1, device
    args = configuration.parser_args()


    # ------------------------------- #
    # Setup args and logs
    # ------------------------------- #
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.model_dir is None:
        print('[Attention]: Please input a model_dir!')
    else:
        args.save_path = f'{args.save_path}/{args.model_dir}'

    # setup log
    log = setup_logger('train_log', args.save_path + '/GCN_training.log')
    result_log = setup_logger('result_log', args.save_path + '/result.log', display=True)

    if args.log_info and not args.debug:
        for key, value in sorted(vars(args).items()):
            log.info(str(key) + ': ' + str(value))
            result_log.info(str(key) + ': ' + str(value))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    # ------------------------------- #
    # Setup GPU device
    # ------------------------------- #
    device = torch.device(f'cuda:{args.gpu}')


    # ------------------------------- #
    # load knowledge graph
    # ------------------------------- #
    knowledge_graph = Graph()
    classFile_to_superclasses, superclassID_to_wikiID =\
        knowledge_graph.class_file_to_superclasses(1, [1,2])
    
    edges = knowledge_graph.edges
    nodes = knowledge_graph.nodes


    # ------------------------------- #
    # Load visual encoder
    # ------------------------------- #
    img_encoder = models.__dict__[args.arch](num_classes=args.num_classes, num_spclasses=len(superclassID_to_wikiID), pool_type=args.pool_type)
    load_checkpoint(img_encoder, 'last')

    if args.log_info and not args.debug:
        log.info("=> loaded checkpoint last")
        result_log.info("=> loaded checkpoint last")


    # ------------------------------- #
    # Init GCN model
    # ------------------------------- #
    layer = 2
    layer_nums = [768, 2048, img_encoder.img_feature_dim]
    layer_nums_str = "".join([str(a)+' ' for a in layer_nums])
    if args.log_info and not args.debug:
        log.info(f'GCN layers: {layer_nums_str}')
        result_log.info(f'GCN layers: {layer_nums_str}')
    GCN = GraphConv.GCN(layer, layer_nums, edges)
    GCN.to(device)
        

    if args.log_info and not args.debug:
        log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in GCN.parameters()])))
        result_log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in GCN.parameters()])))


    # ------------------------------- #
    # define loss function (criterion) and optimizer
    # ------------------------------- #
    criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.SGD(GCN.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    cudnn.benchmark = True


    # ------------------------------- #
    # Load Data Loader
    # ------------------------------- #
    if args.do_meta_train:
        sample_info = [args.meta_train_iter, args.meta_train_way, args.meta_train_shot, args.meta_train_query]
        train_loader, train_label_to_classFile =\
            get_dataloader('train', not args.disable_train_augment, sample=sample_info, spclasses_dict=classFile_to_superclasses)
    else:
        train_loader, train_label_to_classFile =\
            get_dataloader('train', not args.disable_train_augment, shuffle=True, spclasses_dict=classFile_to_superclasses)

    sample_info = [args.meta_val_iter, args.meta_val_way, args.meta_val_shot, args.meta_val_query]
    val_loader, val_label_to_classFile = get_dataloader('val', False, sample=sample_info, spclasses_dict=classFile_to_superclasses)



    # ------------------------------- #
    # Other neccessary parameters
    # ------------------------------- #
    classFile_to_wikiID = miscellaneous.get_classFile_to_wikiID()
    base_cls_index = [nodes.index(classFile_to_wikiID[train_label_to_classFile[i]]) for i in range(len(train_label_to_classFile))]
    val_cls_index = [nodes.index(classFile_to_wikiID[val_label_to_classFile[i]]) for i in range(len(val_label_to_classFile))]

    sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
    desc_embeddings = knowledge_graph.encode_desc(sentence_transformer)
    desc_embeddings =desc_embeddings.to(device)


    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss

    GCN.train()
    start = time.time()

    train_classifiers = get_classifier(img_encoder, img_encoder.img_feature_dim, len(train_label_to_classFile), train_loader, 'train', device, tag='last')
    val_classifiers = get_classifier(img_encoder, img_encoder.img_feature_dim, len(val_label_to_classFile), val_loader, 'val', device, tag='last')

    train_loss_target = torch.ones(train_classifiers.shape[0]).to(device)
    val_loss_target = torch.ones(val_classifiers.shape[0]).to(device)
    
    for epoch in range(50000):

        best_val_cosine_distance = 1000


        base_embeddings = GCN(desc_embeddings)[base_cls_index]

        loss = criterion(base_embeddings, train_classifiers, train_loss_target)
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        batch_time.update(time.time() - start)


        if epoch % 1000 == 0:
            val_embeddings = GCN(desc_embeddings)[val_cls_index]
            val_cosine_distance = evaluate_gcn(val_embeddings, val_classifiers)

            if val_cosine_distance < best_val_cosine_distance:
                best_val_cosine_distance = val_cosine_distance
                torch.save(GCN.state_dict(), f'{model_dir}/gcn_best.pth')

            print(f'[{epoch:3d}/50000]'
                f' batch_time: {batch_time.avg:.2f} loss: {losses.avg:.3f}\tVal cosine distance: {val_cosine_distance:.3f}')  
            batch_time.reset() 
            losses.reset() 
            start = time.time() 



    # train_logger.info("="*60)

# ---------------------------------------------------------------------------------------------------------------------------- #
# Auxiliary Functions
# ---------------------------------------------------------------------------------------------------------------------------- #
def evaluate_gcn(val_embeddings, val_classifiers):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    distance = cos(input1, input2).sum().item()

    return -distance



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


def get_classifier(img_encoder, img_feature_dim, num_classes, data_loader, split, device, tag='last'):
    save_dir = '{}/{}/{}'.format(args.save_path, tag, args.enlarge)
    if os.path.isfile(f'{save_dir}/{split}_classifiers.pkl'):
        classifiers = load_pickle(f'{save_dir}/{split}_classifiers.pkl')
        return classifiers


    img_encoder.eval()
    classifiers = torch.zeros(num_classes, img_feature_dim, dtype=torch.float32)
    with torch.no_grad():
        for _, (imgs, labels, _) in enumerate(tqdm(data_loader)):
            imgs = imgs.to(device)
            img_features, _ = img_encoder(imgs)

            img_features = img_features.to('cpu')
            # # del img_features
            for i, label in enumerate(labels):
                classifiers[label] += img_features[i]
        
    classifiers /= 600
    
    with open(f'{save_dir}/{split}_classifiers.pkl', 'wb') as f:
        pickle.dump(classifiers, f, pickle.HIGHEST_PROTOCOL)

    return classifiers


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
    return loader, sets.label_to_classFile


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
