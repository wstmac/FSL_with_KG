import numpy as np
import os
from config import DATA_PATH, MINI_CLASSES_PATH
import torch
import logging

EPSILON = 1e-8


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_splits():
    background = os.listdir(DATA_PATH + '/miniImageNet/images_background/')
    support = os.listdir(DATA_PATH + '/miniImageNet/images_evaluation/')

    base=np.random.choice(background, 64, replace=False)
    val = set(background) - set(base)

    return base, list(val), support


def get_wikiID_to_classFile():
    wikiID_to_classFile = {}
    with open(MINI_CLASSES_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split(' ')) == 2:
                label, wiki_id = line.split(' ')
                class_file = label.split(':')[0].strip()
                wiki_id = wiki_id.strip()
                wikiID_to_classFile[wiki_id] = class_file
    return wikiID_to_classFile


def get_classFile_to_wikiID():
    classFile_to_wikiID = {}
    with open(MINI_CLASSES_PATH, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line.split(' ')) == 2:
                label, wiki_id = line.split(' ')
                class_file = label.split(':')[0].strip()
                wiki_id = wiki_id.strip()
                classFile_to_wikiID[class_file] = wiki_id
    return classFile_to_wikiID

# ------------------------------- #
# Metric Measurement Function
# ------------------------------- #
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


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        # import ipdb; ipdb.set_trace()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


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


def evaluation(embeddings, labels, n, k, q, distance_metric):
    support = embeddings[:k*n]
    queries = embeddings[k*n:]
    centroids = compute_centroid(support, k, n)


    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = pairwise_distances(queries, centroids, distance_metric)

    accs = accuracy(-distances, labels)

    return accs


def compute_centroid(support, k, n):
    class_centroids = support.reshape(k, n, -1).mean(dim=1)
    return class_centroids


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
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

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