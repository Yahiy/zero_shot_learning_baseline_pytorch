from __future__ import print_function, absolute_import
import os.path as osp
import os
import json
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
import datetime
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.TestSet import TestSet
from utils.transform_test_image import get_test_attrs
from utils.transform_word_embeddings import get_test_embeds
from utils.utils import AverageMeter, load_checkpoint
from model.resnet import ResNet, resnet50
from model.ResNet import ResNet50M

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,'


def get_test_data(data_dir, batch_size):

    root = osp.join(data_dir, 'DatasetB_20180919/test')
    data_file = open(osp.join(data_dir, 'test_list_0919.txt'))
    test_set = data_file.readlines()

    print('test imgs', len(test_set))

    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.Resize(225),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_loader = DataLoader(
        TestSet(test_set, root=root, transform=test_transforms),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    print('get train data est')
    test_cls_list, test_attrs = get_test_embeds()
    return test_loader, test_cls_list, test_attrs


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""
  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
  """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """
  assert type in ['cosine', 'euclidean']
  if type == 'cosine':
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist
  else:
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    dist = np.sqrt(squared_dist)
    return dist


def test(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True

    test_loader, test_cls_list, test_attrs = get_test_data(args.data_dir, args.batch_size)
    model = resnet50(pretrained=False, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=300)
    # model = ResNet50M(num_classes=300)
    print(model)
    model = nn.DataParallel(model).cuda()
    checkpoint = load_checkpoint(osp.join(args.model_dir, 'checkpoint_bi1901_1402.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    feat, name = [], []

    for i,d in enumerate(test_loader):
        imgs, fnames = d
        inputs = Variable(imgs)
        _, outputs , _= model(inputs)
        # outputs = F.sigmoid(outputs)
        feat.append(outputs.data.cpu().numpy())
        name.extend(fnames)
    feat = np.vstack(feat)
    # name = name.hstack(name)
    dist = compute_dist(feat, test_attrs, 'cosine')
    result = []
    for i, v in enumerate(dist):
        max = v.max()
        v = list(v)
        index = v.index(max)
        result.append('{}\tZJL{}'.format(name[i].strip(), test_cls_list[index]))
    print(result[:10])

    s = open('/home/stage/yuan/ZSL/submit_{}.txt'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'w')
    for i in result:
        s.write(i+'\n')
    print(len(result))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=4)

    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')

    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/Dataset')
    parser.add_argument('--model-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/script/model64')

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    test(parser.parse_args())

