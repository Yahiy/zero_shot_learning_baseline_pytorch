from __future__ import absolute_import
import os.path as osp
import numpy as np
from PIL import Image


class TestSet(object):
    def __init__(self, dataset,root=None, transform=None):
        super(TestSet, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

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
        label = int(fname[:3])
        return img, fname, label
