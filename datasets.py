from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import os

from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple

from config import DATA_PATH


class MiniImageNet(Dataset):
    def __init__(self, subset, base_cls, val_cls, support_cls, class_file_to_superclasses, eval = False):
        """Dataset class representing miniImageNet dataset

        # Arguments:
            subset: Whether the dataset represents the background or evaluation set
        """
        if subset not in ('base', 'val', 'support'):
            raise(ValueError, 'subset must be one of (background, evaluation)')
        self.subset = subset
        self.base_cls = base_cls
        self.val_cls = val_cls
        self.support_cls = support_cls
        self.class_file_to_superclasses = class_file_to_superclasses
        self.eval = eval

        self.df = pd.DataFrame(self.index_subset(self.subset))

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Resize(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]

        if self.eval:
            return instance, label
        else:
            #super class label: multi-hot
            class_file = self.df.loc[item]['class_name']
            sp_label = self.class_file_to_superclasses[class_file]
            return instance, label, sp_label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def index_subset(self, subset):
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        print('Indexing {}...'.format(subset))
        # Quick first pass to find total for tqdm bar

        if subset == 'base':
            for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images_background/'):
                class_name = root.split('/')[-1]
                if class_name in self.base_cls:
                    for f in files:
                        images.append({
                            'subset': subset,
                            'class_name': class_name,
                            'filepath': os.path.join(root, f)
                        })
        elif subset == 'val':
            for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images_background/'):
                class_name = root.split('/')[-1]
                if class_name in self.val_cls:
                    for f in files:
                        images.append({
                            'subset': subset,
                            'class_name': class_name,
                            'filepath': os.path.join(root, f)
                        })
        elif subset == 'support':
            for root, _, files in os.walk(DATA_PATH + '/miniImageNet/images_evaluation/'):
                class_name = root.split('/')[-1]
                if class_name in self.support_cls:
                    for f in files:
                        images.append({
                            'subset': subset,
                            'class_name': class_name,
                            'filepath': os.path.join(root, f)
                        })

        return images


class SupportingSetSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 episodes_per_epoch: int = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(SupportingSetSampler, self).__init__(dataset)
        self.dataset = dataset
        self.k = k
        self.n = n
        self.q = q
        self.episodes_per_epoch = episodes_per_epoch

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            df = self.dataset.df
            batch = []
            support_cls = {}
            episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)

            for cls in episode_classes:
                support = df[df['class_id'] == cls].sample(self.n)
                support_cls[cls] = support
                for i, s in support.iterrows():
                    batch.append(s['id'])

            for cls in episode_classes:
                query = df[(df['class_id'] == cls) & (~df['id'].isin(support_cls[cls]['id']))].sample(self.q)
                for i, q in query.iterrows():
                    batch.append(q['id'])

            yield np.stack(batch)


def prepare_nshot_task(n: int, k: int, q: int, data):
    """Typical n-shot task preprocessing.

    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """
    x = data[0].cuda()
    # Create dummy 0-(num_classes - 1) label
    y = torch.arange(0, k, 1 / q).long().cuda()
    return x, y
