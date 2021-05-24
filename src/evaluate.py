import logging
import os
import random
import shutil
import time
import collections
import pickle
import tqdm
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from scipy.stats import mode
from sentence_transformers import SentenceTransformer
import numpy as np


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import datasets
import models
from graph import Graph
from utils import configuration, miscellaneous

# parser = argparse.ArgumentParser()


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    global args
    args = configuration.parser_args()

    args.save_path = f'{args.save_path}/{args.model_dir}'

    result_log = setup_logger('result_log', args.save_path + '/result.log', display=False)

    # for key, value in sorted(vars(args).items()):
    #     result_log.info(str(key) + ': ' + str(value))


    # load sentence transformer
    sentence_transformer = SentenceTransformer('stsb-roberta-large')

    # load knowledge graph
    knowledge_graph = Graph()
    classFile_to_superclasses, superclassID_to_wikiID =\
        knowledge_graph.class_file_to_superclasses(1, [1,2], sentence_transformer, args.gpu)

    model = models.__dict__[args.arch](num_classes=64, sp_embedding_feature_dim = 1024, pool_type='avg_pool', top_k=32)
    model = model.cuda(args.gpu)

    used_files = set(miscellaneous.get_classFile_to_wikiID().keys())
    do_extract_and_evaluate(model, args.model_name, result_log, classFile_to_superclasses, used_files)


def load_checkpoint(model, model_name):
    print(f"=> loading checkpoint {args.save_path}/{model_name}")
    loc = 'cuda:{}'.format(args.gpu)
    checkpoint = torch.load(f'{args.save_path}/{model_name}', map_location=loc)

    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    

    # import ipdb; ipdb.set_trace()
    model.load_state_dict(state_dict)

    print("=> loaded pre-trained model")

    return model



def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


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


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str  = 'l2') -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class centroids. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + miscellaneous.EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + miscellaneous.EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise(ValueError('Unsupported similarity function'))


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



def extract_feature(train_loader, val_loader, model, model_name):
    # return out mean, fcout mean, out feature, fcout features
    saved_feature_name = args.save_path + f'/{model_name.split(".")[0]}_output.plk'

    if os.path.isfile(saved_feature_name):
        data = load_pickle(saved_feature_name)
        return data


    model.eval()
    with torch.no_grad():
        # get training mean
        out_mean, sp_out_mean, fc_out_mean = [], [], []
        for i, (inputs, _, _) in enumerate(warp_tqdm(train_loader)):
            inputs = inputs.cuda(args.gpu)
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
        sp_ground_truth_dict = {}

        for i, (inputs, labels, sp_ground_truth_features) in enumerate(warp_tqdm(val_loader)):
            # compute output
            inputs = inputs.cuda(args.gpu)
            outputs, fc_outputs, sp_outputs, _ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            sp_outputs = sp_outputs.cpu().data.numpy()
            if fc_outputs is not None:
                fc_outputs = fc_outputs.cpu().data.numpy()
            else:
                fc_outputs = [None] * outputs.shape[0]
            for out, sp_out, fc_out, sp_ground_truth_feature, label in zip(outputs, sp_outputs, fc_outputs, sp_ground_truth_features, labels):
                output_dict[label.item()].append([out, sp_out])
                fc_output_dict[label.item()].append(fc_out)

                if label.item() not in sp_ground_truth_dict.keys():
                    sp_ground_truth_dict[label.item()] = sp_ground_truth_feature

        all_info = [out_mean, sp_out_mean, fc_out_mean, output_dict, fc_output_dict, sp_ground_truth_dict]
        save_pickle(saved_feature_name, all_info)
        return all_info


def get_dataloader(split, used_files, aug=False, shuffle=True, out_name=False, sample=None, spclasses_dict=None):
    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(84, disable_random_resize=args.disable_random_resize)
        transform = datasets.TwoCropTransform(transform)
    else:
        transform = datasets.without_augment(84, enlarge=args.enlarge)
    sets = datasets.DatasetFolder(args.data, args.split_dir, split, used_files, transform, out_name=out_name, spclasses_dict=spclasses_dict)

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


def meta_evaluate(data, train_mean, sp_train_mean, sp_ground_truth_dict, shot):
    un_list = []
    l2n_list = []
    cl2n_list = []

    # ----------------------------------------------------------- #
    # Store acc2: get acc without super class information
    # ----------------------------------------------------------- #
    un_list_2 = []
    l2n_list_2 = []
    cl2n_list_2 = []

    sp_acc_list = []

    for _ in warp_tqdm(range(args.meta_test_iter)):
        train_data, test_data, sp_train_data, sp_test_data, train_label, test_label = sample_case(data, shot)
        acc, sp_acc, acc_2 = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
                                train_mean=train_mean, sp_train_mean=sp_train_mean, sp_ground_truth_dict = sp_ground_truth_dict, norm_type='CL2N')
        cl2n_list.append(acc)
        cl2n_list_2.append(acc_2)
        sp_acc_list.append(sp_acc)
        acc, _, acc_2 = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
                                train_mean=train_mean, sp_train_mean=sp_train_mean, sp_ground_truth_dict = sp_ground_truth_dict, norm_type='L2N')
        l2n_list.append(acc)
        l2n_list_2.append(acc_2)
        acc, _, acc_2 = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
                                train_mean=train_mean, sp_train_mean=sp_train_mean, sp_ground_truth_dict = sp_ground_truth_dict, norm_type='UN')
        un_list.append(acc)
        un_list_2.append(acc_2)
        # acc = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
        #                         train_mean=train_mean, sp_train_mean=sp_train_mean, sp_ground_truth_dict = sp_ground_truth_dict, norm_type='CL2N')
        # cl2n_list.append(acc)
        # acc = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
        #                         train_mean=train_mean, sp_train_mean=sp_train_mean, sp_ground_truth_dict = sp_ground_truth_dict, norm_type='L2N')
        # l2n_list.append(acc)
        # acc = metric_class_type(train_data, test_data, sp_train_data, sp_test_data, train_label, test_label, shot,
        #                         train_mean=train_mean, sp_train_mean=sp_train_mean, sp_ground_truth_dict = sp_ground_truth_dict, norm_type='UN')
        # un_list.append(acc)
    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    sp_mean, sp_conf = compute_confidence_interval(sp_acc_list)

    un_mean_2, un_conf_2 = compute_confidence_interval(un_list_2)
    l2n_mean_2, l2n_conf_2 = compute_confidence_interval(l2n_list_2)
    cl2n_mean_2, cl2n_conf_2 = compute_confidence_interval(cl2n_list_2)
    return un_mean, un_conf, l2n_mean, l2n_conf, cl2n_mean, cl2n_conf, sp_mean, sp_conf, un_mean_2, un_conf_2, l2n_mean_2, l2n_conf_2, cl2n_mean_2, cl2n_conf_2


def metric_class_type(gallery, query, sp_gallery, sp_query, train_label, test_label, shot, train_mean=None, sp_train_mean=None, sp_ground_truth_dict = None, norm_type='CL2N'):
    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]

    elif norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    gallery = gallery.reshape(args.meta_val_way, shot, gallery.shape[-1]).mean(1)

    sp_gallery = np.zeros((args.meta_val_way, 1024))
    for i, l in enumerate(list(dict.fromkeys(train_label))):
        sp_gallery[i] = sp_ground_truth_dict[l].numpy()

    train_label = train_label[::shot]
    test_label = np.array(test_label)
    subtract = gallery[:, None, :] - query

    distance = LA.norm(subtract, 2, axis=-1)   
    sp_distance = cosine_similarity(sp_gallery, sp_query)

    sp_candidates = max(3, int(args.meta_val_way * 0.25)) # select candidates according to the super features
    sp_idx = np.argpartition(sp_distance, sp_candidates, axis=0)[:sp_candidates]

    sp_nearest_samples = np.take(train_label, sp_idx)
    sp_acc = 0
    for i in range(sp_idx.shape[1]):
        if test_label[i] in sp_nearest_samples[:,i]:
            sp_acc += 1
    sp_acc = sp_acc / sp_idx.shape[1]

    restricted_distance = np.ones(distance.shape) * 100
    for i in range(sp_idx.shape[1]):                                                                                                                                                                             
            restricted_distance[sp_idx[:,i],i] = distance[sp_idx[:,i],i] 


    idx = np.argpartition(restricted_distance, args.num_NN, axis=0)[:args.num_NN]
    nearest_samples = np.take(train_label, idx)
    out = mode(nearest_samples, axis=0)[0]
    out = out.astype(int)
    
    acc = (out == test_label).mean()



    # ----------------------------------------------------------- #
    # Compute acc without super class information
    # ----------------------------------------------------------- #
    idx_2 = np.argpartition(distance, args.num_NN, axis=0)[:args.num_NN]
    nearest_samples_2 = np.take(train_label, idx_2)
    out_2 = mode(nearest_samples_2, axis=0)[0]
    out_2 = out_2.astype(int)
    # test_label = np.array(test_label)
    acc_2 = (out_2 == test_label).mean()

    return acc, sp_acc, acc_2




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


def do_extract_and_evaluate(model, model_name, log, classFile_to_superclasses, used_files):
    train_loader = get_dataloader('train', used_files, aug=False, shuffle=False, out_name=False, spclasses_dict=classFile_to_superclasses)
    val_loader = get_dataloader('test', used_files, aug=False, shuffle=False, out_name=False, spclasses_dict=classFile_to_superclasses)
    # import ipdb; ipdb.set_trace()
    load_checkpoint(model, model_name)
    out_mean, sp_out_mean, fc_out_mean, out_dict, fc_out_dict, sp_ground_truth_dict = extract_feature(train_loader, val_loader, model, model_name)
    accuracy_info_shot1 = meta_evaluate(out_dict, out_mean, sp_out_mean, sp_ground_truth_dict, 1)
    accuracy_info_shot5 = meta_evaluate(out_dict, out_mean, sp_out_mean, sp_ground_truth_dict, 5)

    sp_candidates = max(3, int(args.meta_val_way * 0.25))
    if args.log_info and not args.debug:
        log.info(f'============{model_name}============')
        log.info(
            'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\tSP_ACC({}\{})\tUN_2\tL2N_2\tCL2N_2\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
                sp_candidates, args.meta_val_way, 'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))
    else:
        print(
        'Meta Test: LAST\nfeature\tUN\tL2N\tCL2N\tSP_ACC({}\{})\tUN_2\tL2N_2\tCL2N_2\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\n{}\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})\t{:.4f}({:.4f})'.format(
                sp_candidates, args.meta_val_way, 'GVP 1Shot', *accuracy_info_shot1, 'GVP_5Shot', *accuracy_info_shot5))


if __name__ == '__main__':
    main()
