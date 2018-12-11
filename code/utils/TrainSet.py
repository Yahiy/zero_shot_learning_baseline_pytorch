from __future__ import absolute_import
import os.path as osp
import numpy as np
from PIL import Image


class TrainSet(object):
    def __init__(self, dataset, attrs, labels,root=None, transform=None):
        super(TrainSet, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.attrs = attrs
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname.strip())
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        cls_name = fname[:4]
        attr = self.attrs[cls_name]
        attr2 = map(np.float32, attr)
        label = self.labels[cls_name]
        return img, fname, attr2, int(label)

