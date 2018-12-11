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
import visdom
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.TrainSet import TrainSet
from utils.TestSetforTrain import TestSet
from utils.transform_test_image import get_val_attrs
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from model.resnet import ResNet, resnet50
from model.Resnet import get_network
from test import compute_dist
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

vis = visdom.Visdom(env='ZSL-64')
vis.check_connection()


def get_data(data_dir, batch_size):

    root_train = osp.join(data_dir, 'train')
    attrs_file = open(osp.join(data_dir, 'DatasetA/attrs.json'))
    cls_file = open(osp.join(data_dir, 'DatasetA/cls.json'))
    data_file = open(osp.join(data_dir, 'meta.json'))
    attrs = json.load(attrs_file)
    cls_labels = json.load(cls_file)
    data_set = json.load(data_file)
    train_set = data_set['images']
    print('train imgs', len(train_set))

    root_test = osp.join(data_dir, 'DatasetA/test')
    val_cls_list, val_attrs = get_val_attrs()

    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.Resize(225),
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.Resize(225),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        TrainSet(train_set, attrs=attrs, labels=cls_labels, root=root_train, transform=train_transforms),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TestSet(val_cls_list, root=root_test, transform=test_transforms),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    # val_cls_list, val_attrs = get_val_attrs()
    print('get train data loader')
    return train_loader, val_loader, val_cls_list, val_attrs


def valid(val_loader, val_cls_list, val_attrs, model_dir):
    # model = resnet50(pretrained=False, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=30)
    model = get_network(num_classes=30, depth=50)
    model = nn.DataParallel(model).cuda()

    checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint_64_21.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    feat, name = [], []

    for i, d in enumerate(val_loader):
        imgs, fnames, _ = d
        inputs = Variable(imgs)
        _, outputs = model(inputs)
        # outputs = F.sigmoid(outputs)
        feat.append(outputs.data.cpu().numpy())
        name.extend(fnames)
    feat = np.vstack(feat)
    # name = name.hstack(name)
    dist = compute_dist(feat, val_attrs, 'cosine')
    result = []
    for i, v in enumerate(dist):
        max = v.max()
        v = list(v)
        index = v.index(max)
        result.append((int(name[i][:3]), val_cls_list[index]))
    n = 0
    for pre, tar in result:
        if pre == tar:
            n = n+1
    print('the acc is {}/{}'.format(n, len(result)))


def main(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    train_loader, val_loader, val_cls_list, val_attrs = get_data(args.data_dir, args.batch_size)
    model = resnet50(pretrained=True, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=30)
    # model = get_network(num_classes=30, depth=50)
    print(model)
    model = nn.DataParallel(model).cuda()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    def adjust_lr(epoch):
        if epoch in [80]:
            lr = 0.1 * args.lr
            print('=====> adjust lr to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # test(test_loader, test_cls_list, test_attrs, args.model_dir)

    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.train()

        loss = AverageMeter()
        iteration  = 467 * epoch
        # print(iteration)

        for i,d in enumerate(train_loader):
            iteration += 1

            imgs, fnames, attrs, labels = d
            inputs = Variable(imgs)
            attr_targets = Variable(torch.stack(attrs, 1).cuda())
            attr_targets = attr_targets.detach()
            _, outputs = model(inputs)
            mse_loss = nn.MultiLabelSoftMarginLoss(size_average=False)(outputs, attr_targets)
            mse_loss = mse_loss/(8*args.batch_size)
            loss.update(mse_loss.data[0], attr_targets.size(0))
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss.avg,)),
                     win='reid softmax loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss', legend=['Loss'])
                     )

            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t Loss {:.6f} ({:.6f})\t'
                      .format(epoch, i + 1, len(train_loader),
                              loss.val, loss.avg))

            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': 0,
            }, False, fpath=osp.join(args.model_dir, 'checkpoint_64_21.pth.tar'))
        valid(val_loader, val_cls_list, val_attrs, args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('-b', '--batch-size', type=int, default=64)
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

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/Dataset')
    parser.add_argument('--model-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/script/model')

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

