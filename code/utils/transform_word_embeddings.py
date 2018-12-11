from __future__ import print_function
import sys
sys.path.insert(0, '.')

import os.path as osp
import os
import shutil
import json
import numpy as np
from scipy.misc import imsave


def convert_cls_word():
    word_file = open('../data/DatasetB/label_list.txt')
    dict = {}
    for w in word_file.readlines():
        cls, word = w.strip().split('\t')
        dict[word] = '{:04d}'.format(int(cls[3:]))
    return dict


def get_word_embed():
    word_to_cls = convert_cls_word()
    embed_file = open('../data/DatasetB/class_wordembeddings.txt')
    dict = {}
    for e in embed_file.readlines():
        d = e.strip().split(' ')
        word = d[0]
        cls = word_to_cls[word]
        embed = []
        for i in d[1:]:
            embed.append(np.float32(i))
        dict[cls] = embed
    return dict


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


def get_test_cls(test_file, test_file_old):
    train_cls = get_train_cls('../data/train_list.txt')
    valid_cls = get_val_cls('../data/DatasetA/submit.txt')

    src_imgs = open(test_file, 'r')
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

    src_imgs_old = open(test_file_old, 'r')
    test_cls_old = []
    cls_num_old = 0
    for d in src_imgs_old.readlines():
        cls, name = d.strip().split('	')
        cls = int(cls[3:])
        if cls not in train_cls and cls not in valid_cls:
            test_cls_old.append(cls)
            cls_num_old = cls_num_old + 1
    print(cls_num_old)
    print(test_cls_old)

    test_B = []
    for i in test_cls:
        if i not in test_cls_old:
            test_B.append(i)
    print(test_B)
    # test_cls = test_cls[1:]
    return test_B


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
    write_json(attrs, '../data/DatasetA/test_attrs.json')


def get_test_embeds():
    test_cls = get_test_cls('../data/DatasetBlabel_list.txt', '../data/DatasetA/label_list.txt')
    embed = get_word_embed()
    attrs = []
    for t in test_cls:
        t = '{:04d}'.format(int(t))
        e = embed[str(t)]
        attrs.append(e)
    print('test:', test_cls)
    return test_cls, np.array(attrs)


def get_val_attrs():
    val_cls = get_val_cls('../data/DatasetA/submit.txt')
    embed = get_word_embed()
    attrs = []
    for val in val_cls:
        val = '{:04d}'.format(int(val))
        e = embed[val]
        attrs.append(e)
    print('val:', val_cls)
    return val_cls, np.array(attrs)


if __name__ == '__main__':
    # get_test_cls('../Dataset/DatasetA_test/label_list.txt')
    # save_attributes('../Dataset/DatasetA_test/attributes_per_class.txt')
    # get_test_attrs()
    print()