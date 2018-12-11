from __future__ import print_function
import sys
sys.path.insert(0, '.')

import os.path as osp
import os
import shutil
import json
import numpy as np
from scipy.misc import imsave

from utils import mkdir_if_missing


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_image(train_file):
    src_imgs = open(train_file, 'r')
    n,c,cls_num = 0, -1, 0
    new_im_names = []
    cls_dict = dict()
    for d in src_imgs.readlines():
        img, cls = d.strip().split('	')
        img_path = osp.join('../data/DatasetA/train', img)
        cls = int(cls[3:])
        if cls != c:
            c = cls
            n = 0
            cls_dict[str(cls)] = cls_num
            cls_num = cls_num +1
            print(cls)
        new_im_name = '{:04d}_0918_{:04d}.jpeg'.format(cls, n)
        shutil.copy(img_path, osp.join('../data/train', new_im_name))
        new_im_names.append(new_im_name)
        n = n+1

    meta = {'images': new_im_names}
    write_json(meta, osp.join('../data', 'meta_0919.json'))
    # write_json(cls_dict, osp.join('../Dataset/DatasetA', 'cls.json'))


def get_train_cls(train_file):
    src_imgs = open(train_file, 'r')
    cls_list = []
    c,cls_num = 0, 0
    for d in src_imgs.readlines():
        cls = d.strip().split('_')
        cls = int(cls[0])
        if cls != c and cls not in cls_list:
            c = cls
            cls_list.append(cls)
            cls_num = cls_num +1
    print(cls_num)
    print(cls_list)
    cls_list = map(int, cls_list)
    cls_list.sort()
    cls_js = dict()
    for i, cls in enumerate(cls_list):
        key = '{:04d}'.format(cls)
        cls_js[key] = i
    # write_json(cls_js, osp.join('../Dataset', 'cls_0919.json'))
    return cls_list


def save_attributes(attr_file):
    src_file = open(attr_file, 'r')
    attrs = dict()
    for d in src_file.readlines():
        d = d.strip().split('	')
        if len(d) == 31:
            key = d[0][3:]
        else:
            key = d[0]
            print(key)
        attr = []
        for i in d[1:]:
            attr.append(i)
        attrs[key] = attr
    write_json(attrs, '../Dataset/DatasetA/attrs.json')


if __name__ == '__main__':
    # save_image('../Dataset/DatasetA/train.txt')
    # save_attributes('../Dataset/DatasetA/attributes_per_class.txt')
    get_train_cls('../Dataset/train_list_0919.txt')
    print()