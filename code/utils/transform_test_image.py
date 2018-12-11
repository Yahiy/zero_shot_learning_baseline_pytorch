from __future__ import print_function
import sys
sys.path.insert(0, '.')

import os.path as osp
import os
import shutil
import json
import numpy as np
from scipy.misc import imsave


def get_val_cls(train_file):
    src_imgs = open(train_file, 'r')
    cls_list = []
    c,cls_num = 0, 0
    for d in src_imgs.readlines():
        img, cls = d.strip().split('	')
        cls = int(cls[3:])
        if cls != c and cls not in cls_list:
            c = cls
            cls_list.append(cls)
            cls_num = cls_num +1
    print(cls_num)
    cls_list.sort()
    print(cls_list)
    return cls_list


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

    # cls_list = map(int, cls_list)
    cls_list.sort()
    print(cls_num)
    print(cls_list)
    # cls_js = dict()
    # for i, cls in enumerate(cls_list):
    #     key = '{:04d}'.format(cls)
    #     cls_js[key] = i
    # write_json(cls_js, osp.join('../Dataset', 'cls_0919.json'))
    return cls_list


def get_test_cls(test_file):
    train_cls = get_train_cls('../Dataset/train_list_0919.txt')
    valid_cls = get_val_cls('../Dataset/DatasetA/submit.txt')
    src_imgs = open(test_file, 'r') # label file
    test_cls = []
    cls_num = 0
    for d in src_imgs.readlines():
        cls, name = d.strip().split('	')
        cls = int(cls[3:])
        if cls not in train_cls and cls not in valid_cls:
            test_cls.append(cls)
            cls_num = cls_num +1
    print(cls_num)
    print(test_cls)
    return test_cls


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
    write_json(attrs, '../Dataset/DatasetA_test/test_attrs.json')


def get_test_attrs():
    test_cls = get_test_cls('../Dataset/DatasetB_20180919/label_list.txt')
    src_file = open('../Dataset/DatasetB_20180919/attributes_per_class.txt', 'r')
    test_list, attrs = [], []
    for d in src_file.readlines():
        d = d.strip().split('	')
        if len(d) == 31:
            key = d[0][3:]
        else:
            key = d[0]
            print(key)
        key = int(key)
        if key in test_cls:
            test_list.append(key)
            attr = []
            for i in d[1:]:
                attr.append(np.float32(i))
            attrs.append(attr)
    print('test:',test_list)
    return test_list, np.array(attrs)


def get_val_attrs():
    val_cls = get_train_cls('../Dataset/DatasetA/submit.txt')
    src_file = open('../Dataset/DatasetA_test/attributes_per_class.txt', 'r')
    val_list, attrs = [], []
    for d in src_file.readlines():
        d = d.strip().split('	')
        if len(d) == 31:
            key = d[0][3:]
        else:
            key = d[0]
            print(key)
        key = int(key)
        if key in val_cls:
            val_list.append(key)
            attr = []
            for i in d[1:]:
                attr.append(np.float32(i))
            attrs.append(attr)
    print('val:', val_list)
    return val_list, np.array(attrs)


if __name__ == '__main__':
    # get_test_cls('../Dataset/DatasetA_test/label_list.txt')
    # save_attributes('../Dataset/DatasetA_test/attributes_per_class.txt')
    get_test_attrs()
    print()