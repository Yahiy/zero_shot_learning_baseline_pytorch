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
from utils.ValSet import ValSet
# from utils.transform_test_image import get_val_attrs
from utils.transform_word_embeddings import get_val_attrs, get_word_embed
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from model.resnet import ResNet, resnet50, normalize
from model.Resnet import get_network
from model.ResNet import ResNet50M
from test import compute_dist
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

vis = visdom.Visdom(env='ZSL-decode2')
vis.check_connection()


def get_mse_loss(input,target,batch):
    input_pred = input
    target_pred = target.detach()
    # input_pred = F.normalize(input_pred)
    # target_pred = F.normalize(target_pred)
    mse = nn.MSELoss(size_average=False)(input_pred, target_pred)
    loss = mse / (8*batch)
    return loss


def get_data(data_dir, batch_size):
    root_train = osp.join(data_dir, 'train')
    # attrs_file = open(osp.join(data_dir, 'DatasetA/attrs.json'))
    cls_file = open(osp.join(data_dir, 'cls_0919.json'))
    train_file = open(osp.join(data_dir, 'train_list_0919.txt'))
    # attrs = json.load(attrs_file)
    cls_labels = json.load(cls_file)
    # data_set = json.load(data_file)
    train_set = train_file.readlines()
    print('train imgs', len(train_set))

    root_test = osp.join(data_dir, 'DatasetA/test')
    val_file = open(osp.join(data_dir, 'DatasetA/submit.txt'), 'r')
    val_set = val_file.readlines()
    # val_cls_list, val_attrs = get_val_attrs()
    val_cls_list, val_attrs = get_val_attrs()
    attrs = get_word_embed()

    attr_mat = torch.zeros((164, 300))
    for key in cls_labels:
        a = attrs[key]
        a = map(float, a)
        attr_mat[cls_labels[key]] = torch.Tensor(a)

    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.Resize(225),
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.Resize(225),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        TrainSet(train_set, attrs=attrs, labels=cls_labels, root=root_train, transform=train_transforms),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        ValSet(val_set, root=root_test, transform=test_transforms),
        batch_size=batch_size/2, shuffle=False, pin_memory=True
    )
    # val_cls_list, val_attrs = get_val_attrs()
    print('get train data loader')
    return train_loader, val_loader, val_cls_list, val_attrs, attr_mat


def valid(val_loader, val_cls_list, val_attrs, model_dir):
    model = resnet50(pretrained=False, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=300)
    # model = get_network(num_classes=30, depth=50)
    # model = ResNet50M(num_classes=300)
    model = nn.DataParallel(model).cuda()

    checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint_bi_1901.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    feat, label = [], []

    for i, d in enumerate(val_loader):
        imgs, _, l = d
        inputs = Variable(imgs)
        _, outputs, _ = model(inputs)
        outputs = F.relu(outputs)
        feat.append(outputs.data.cpu().numpy())
        label.extend(l)
    feat = np.vstack(feat)
    # name = name.hstack(name)
    dist = compute_dist(feat, val_attrs, 'cosine')
    result = []
    for i, v in enumerate(dist):
        max = v.max()
        v = list(v)
        index = v.index(max)
        result.append((int(label[i]), val_cls_list[index]))
    n = 0
    for tar, pre in result:
        if pre == tar:
            n = n+1
    print('the acc by cos is {}/{}'.format(n, len(result)))
    return n


def train(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    train_loader, val_loader, val_cls_list, val_attrs, attr_mat = get_data(args.data_dir, args.batch_size)
    model = resnet50(pretrained=True, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=300)
    # model = get_network(num_classes=30, depth=50)
    # model = ResNet50M(num_classes=300)
    print(model)
    model = nn.DataParallel(model).cuda()
    #
    # checkpoint = load_checkpoint(osp.join(args.model_dir, 'checkpoint_64.pth.tar'))
    # model.module.load_state_dict(checkpoint['state_dict'])

    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    def adjust_lr(epoch):
        if epoch in [70,]:
            lr = 0.1 * args.lr
            print('=====> adjust lr to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # valid(val_loader, val_cls_list, val_attrs, args.model_dir)
    attr_mat = Variable(attr_mat.t().cuda())
    attr_mat = normalize(attr_mat, axis=1)
    t = [0.01,0.01, 0.002,0.001,0.001,0.001,0.001,0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0001, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.train()

        losses = AverageMeter()
        loss1 = AverageMeter()
        loss2 = AverageMeter()
        iteration  = 116 * epoch
        # print(iteration)

        for i,d in enumerate(train_loader):
            iteration += 1

            imgs, fnames, attrs, labels = d
            inputs = Variable(imgs)
            labels = Variable(labels.cuda())
            attr_targets = Variable(torch.stack(attrs, 1).cuda())
            attr_targets = attr_targets.detach()
            f, outputs, w = model(inputs)

            f_ = torch.mm(attr_targets, w)

            mse_loss = get_mse_loss(f, f_, args.batch_size)
            loss1.update(mse_loss.data[0], attr_targets.size(0))

            cls_output = torch.mm(outputs, attr_mat)
            cls_loss = criterion(cls_output, labels)
            loss2.update(cls_loss.data[0], attr_targets.size(0))

            # if epoch < 5:
            #     loss = cls_loss
            # else:
            loss = t[epoch]*mse_loss + cls_loss
            losses.update(loss.data[0], attr_targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((losses.avg,)),
                     win='loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss', legend=['Loss'])
                     )

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss1.avg,)),
                     win='sigmod loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss1', legend=['Loss'])
                     )

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss2.avg,)),
                     win='biliner loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss2', legend=['Loss'])
                     )

            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t Loss {:.6f} ({:.6f})\t Loss_ml {:.6f} ({:.6f})\t Loss_cl {:.6f} ({:.6f})\t'
                      .format(epoch, i + 1, len(train_loader),
                              losses.val, losses.avg, loss1.val, loss1.avg, loss2.val, loss2.avg))

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint_bi_1901.pth.tar'))
        nu = valid(val_loader, val_cls_list, val_attrs, args.model_dir)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint_bi1901_{}.pth.tar'.format(nu)))

        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((nu,)),
                 win='acc',
                 update='append' if iteration > 0 else None,
                 opts=dict(xlabel='epoch', title='acc', legend=['acc'])
                 )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('-b', '--batch-size', type=int, default=256)
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
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--epochs', type=int, default=150)
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
    train(parser.parse_args())

