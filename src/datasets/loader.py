import os

import PIL.Image as Image
import numpy as np

__all__ = ['DatasetFolder']


class DatasetFolder(object):

    def __init__(self, root, split_dir, split_type, used_files, transform, out_name=False, spclasses_dict=None):
        assert split_type in ['train', 'test', 'val', 'query', 'repr']
        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '' and x.strip().split(',')[1] in used_files]

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]
        # import ipdb; ipdb.set_trace()
        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.data)
        self.label_to_classFile = label_key
        self.class_file_to_superclasses = spclasses_dict

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert os.path.isfile(self.root+'/'+self.data[index])
        img = Image.open(self.root + '/' + self.data[index]).convert('RGB')
        label = self.labels[index]
        label = int(label)

        img = self.transform(img)

        if self.class_file_to_superclasses != None:
            # get super label
            class_file = self.data[index][:9]
            sp_label = self.class_file_to_superclasses[class_file]

            return img, label, sp_label
        else:
            return img, label

