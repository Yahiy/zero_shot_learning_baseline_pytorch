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
from model.resnet import ResNet, resnet50
from model.Resnet import get_network
from model.ResNet import ResNet50M
from test import compute_dist
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

vis = visdom.Visdom(env='ZSL-ml')
vis.check_connection()


def mse_loss(input,target,batch):
    input_pred = input
    target_pred = target.detach()
    input_pred = F.normalize(input_pred)
    target_pred = F.normalize(target_pred)
    mse = nn.MSELoss(size_average=False)(input_pred, target_pred)
    loss = mse / (1*batch)
    return loss


def get_data(data_dir, batch_size):

    root_train = osp.join(data_dir, 'train')
    # attrs_file = open(osp.join(data_dir, 'DatasetA/attrs.json'))
    cls_file = open(osp.join(data_dir, 'DatasetA/cls.json'))
    data_file = open(osp.join(data_dir, 'meta.json'))
    # attrs = json.load(attrs_file)
    cls_labels = json.load(cls_file)
    data_set = json.load(data_file)
    train_set = data_set['images']
    print('train imgs', len(train_set))

    root_test = osp.join(data_dir, 'DatasetA/test')
    val_file = open(osp.join(data_dir, 'DatasetA/submit.txt'), 'r')
    val_set = val_file.readlines()
    # val_cls_list, val_attrs = get_val_attrs()
    val_cls_list, val_attrs = get_val_attrs()
    attrs = get_word_embed()

    attr_mat = torch.zeros((149,300))
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


def valid(val_loader, val_cls_list, val_attrs, model_dir, pth):
    # model = resnet50(pretrained=False, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=30)
    # model = get_network(num_classes=30, depth=50)
    model = ResNet50M(num_classes=300)
    model = nn.DataParallel(model).cuda()

    checkpoint = load_checkpoint(osp.join(model_dir, pth))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    feat, label = [], []

    for i, d in enumerate(val_loader):
        imgs, _, l = d
        inputs = Variable(imgs)
        _, outputs = model(inputs)
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

    dist = compute_dist(feat, val_attrs, 'euclidean')
    result = []
    for i, v in enumerate(dist):
        min = v.min()
        v = list(v)
        index = v.index(min)
        result.append((int(label[i]), val_cls_list[index]))
    n2 = 0
    for tar, pre in result:
        if pre == tar:
            n2 = n2 + 1
    print('the acc by el is {}/{}'.format(n2, len(result)))
    return n


def main(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    train_loader, val_loader, val_cls_list, val_attrs, attr_mat = get_data(args.data_dir, args.batch_size)
    # model = resnet50(pretrained=True, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=30)
    # model = get_network(num_classes=30, depth=50)
    model_1 = ResNet50M(num_classes=300)
    model_2 = ResNet50M(num_classes=300)
    print(model_1)
    model_1 = nn.DataParallel(model_1).cuda()
    model_2 = nn.DataParallel(model_2).cuda()
    #
    # checkpoint = load_checkpoint(osp.join(args.model_dir, 'checkpoint_64.pth.tar'))
    # model.module.load_state_dict(checkpoint['state_dict'])

    # Optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=args.lr,
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
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model_1.train()
        model_2.train()

        losses_1 = AverageMeter()
        losses_2 = AverageMeter()
        loss_cls1 = AverageMeter()
        loss_cls2 = AverageMeter()
        loss_mse1 = AverageMeter()
        loss_mse2 = AverageMeter()
        iteration  = 116 * epoch
        # print(iteration)

        for i,d in enumerate(train_loader):
            iteration += 1

            # train data and label
            imgs, fnames, attrs, labels = d
            inputs = Variable(imgs)
            labels = Variable(labels.cuda())
            attr_targets = Variable(torch.stack(attrs, 1).cuda())
            attr_targets = attr_targets.detach()

            # output
            _, outputs_1 = model_1(inputs)
            _, outputs_2 = model_2(inputs)
            cls_output_1 = torch.mm(outputs_1, attr_mat)
            cls_output_2 = torch.mm(outputs_2, attr_mat)

            # loss1
            cls_loss_1 = criterion(cls_output_1, labels)
            mse_loss_1 = mse_loss(outputs_1, outputs_2, args.batch_size)
            loss1 = cls_loss_1 + mse_loss_1
            # update model1
            optimizer_1.zero_grad()
            loss1.backward()
            optimizer_1.step()

            # loss2 and update model2
            _, outputs_1 = model_1(inputs)
            cls_loss_2 = criterion(cls_output_2, labels)
            mse_loss_2 = mse_loss(outputs_1, outputs_2, args.batch_size)
            loss2 = cls_loss_2 + mse_loss_2
            optimizer_2.zero_grad()
            loss2.backward()
            optimizer_2.step()

            losses_1.update(loss1.data[0], attr_targets.size(0))
            losses_2.update(loss2.data[0], attr_targets.size(0))
            loss_cls1.update(cls_loss_1.data[0], attr_targets.size(0))
            loss_cls2.update(cls_loss_2.data[0], attr_targets.size(0))
            loss_mse1.update(mse_loss_1.data[0], attr_targets.size(0))
            loss_mse2.update(mse_loss_2.data[0], attr_targets.size(0))

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((losses_1.avg,)),
                     win='loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss1', legend=['Loss'])
                     )

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((losses_2.avg,)),
                     win='loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss2', legend=['Loss'])
                     )

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss_cls1.avg,)),
                     win='sigmod loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='SoftLoss1', legend=['Loss'])
                     )

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss_cls2.avg,)),
                     win='sigmod loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='SoftLoss2', legend=['Loss'])
                     )

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss_mse1.avg,)),
                     win='biliner loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='MSELoss1', legend=['Loss'])
                     )
            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss_mse1.avg,)),
                     win='biliner loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='MSELoss2', legend=['Loss'])
                     )

            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t '
                      'Loss1 {:.6f} ({:.6f})\t Loss_cls1 {:.6f} ({:.6f})\t Loss_mse1 {:.6f} ({:.6f})\t'
                      'Loss2 {:.6f} ({:.6f})\t Loss_cls2 {:.6f} ({:.6f})\t Loss_mse2 {:.6f} ({:.6f})\t'
                      .format(epoch, i + 1, len(train_loader),
                              losses_1.val, losses_1.avg, loss_cls1.val, loss_cls1.avg, loss_mse1.val, loss_mse1.avg,
                              losses_2.val, losses_2.avg, loss_cls2.val, loss_cls2.avg, loss_mse2.val, loss_mse2.avg))

        save_checkpoint({
            'state_dict': model_1.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint_ml_1.pth.tar'))
        save_checkpoint({
            'state_dict': model_2.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint_ml_2.pth.tar'))

        nu_1 = valid(val_loader, val_cls_list, val_attrs, args.model_dir, 'checkpoint_ml_1.pth.tar')
        nu_2 = valid(val_loader, val_cls_list, val_attrs, args.model_dir, 'checkpoint_ml_2.pth.tar')

        save_checkpoint({
            'state_dict': model_1.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint_ml1_{}.pth.tar'.format(nu_1)))
        save_checkpoint({
            'state_dict': model_2.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': 0,
        }, False, fpath=osp.join(args.model_dir, 'checkpoint_ml2_{}.pth.tar'.format(nu_2)))

        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((nu_1,)),
                 win='acc',
                 update='append' if iteration > 0 else None,
                 opts=dict(xlabel='epoch', title='acc1', legend=['acc'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((nu_2,)),
                 win='acc',
                 update='append' if iteration > 0 else None,
                 opts=dict(xlabel='epoch', title='acc2', legend=['acc'])
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
    parser.add_argument('--model-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/script/model_ml')

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

