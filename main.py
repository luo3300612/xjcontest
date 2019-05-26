import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
from tensorboardX import SummaryWriter
import sys
import torch.nn.functional as F
from utils import OutPutUtil
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from random import shuffle
from torchvision.models import resnet18
from pathlib import Path
import pandas as pd
from model import ResNet20
import multiprocessing


class DataSpliter:
    paths = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

    def __init__(self, img_dir, ratio=[0.8, 0.1, 0.1]):
        self.train = []
        self.val = []
        self.test = []

        for path in self.paths:
            img_path = Path(img_dir, path)
            imgs = list(img_path.iterdir())
            num_imgs = len(imgs)
            shuffle(imgs)
            for i in range(0, int(ratio[0] * num_imgs)):
                self.train.append(str(imgs[i]))
            for i in range(int(ratio[0] * num_imgs), int((ratio[0] + ratio[1]) * num_imgs)):
                self.val.append(str(imgs[i]))
            for i in range(int((ratio[0] + ratio[1]) * num_imgs), num_imgs):
                self.test.append(str(imgs[i]))


class Data(Dataset):

    def __init__(self, img_path, visit_path, train=True, transforms=None):
        self.img_path = Path(img_path)
        self.visit_path = visit_path
        self.img_files = list(self.img_path.iterdir())
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img = Image.open(img_file)
        feature = np.load(self.visit_path + img_file.name.split('.')[0] + '.npy')

        if self.transforms is not None:
            img = self.transforms(img)

        sample = {}
        if self.train:
            target = self.csv.iloc[idx][2] - 1
            sample['img'] = img
            sample['feature'] = feature
            sample['target'] = target
        else:
            sample['img'] = img
            sample['feature'] = feature
        return sample


def adjust_learning_rate(optimizer, iteration, n_iter, init_lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iteration / n_iter <= 0.5:
        lr = init_lr
    elif 0.5 < iteration / n_iter <= 0.75:
        lr = init_lr / 10
    else:
        lr = init_lr / 100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
        self.img_feature_extracter = nn.Sequential(*list(resnet18().children())[:-1])
        self.visit_feature_extracter = ResNet20(in_channel=7)
        self.fc = nn.Linear(512 + 64, num_classes)

    def forward(self, img, text):
        img_feature = self.feature_extracter(img)
        visit_feature = self.visit_feature_extracter(text)
        out = torch.cat((img_feature, visit_feature), dim=1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Em blah blah blah')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training and testing (default: 64)')
    parser.add_argument('--n-iter', type=int, default=64000,
                        help='n-iter (default: 64000)')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='val interveal (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-path', type=str, default='./train.log',
                        help='path to save log file (default: ./train.log)')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use adam')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum (default: 0)')
    parser.add_argument('--save-path', type=str, default='./result',
                        help='save path (default: ./result)')
    parser.add_argument('--val', action='store_true', default=False,
                        help='validation mode')
    parser.add_argument('--comment', type=str, default='',
                        help='comment')
    args = parser.parse_args()
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    monitor = OutPutUtil(True, True, args.log_path)
    monitor.speak(args)
    batch_size = args.batch_size

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    monitor.speak("Split Data ...")

    spliter = DataSpliter('/userhome/bigdata/train')
    train_path = spliter.train
    val_path = spliter.val
    test_path = spliter.test

    visit_dir = '/userhome/bigdata/train/visit_feat'

    monitor.speak("OK\ntrain:{},val:{},test:{}".format(len(train_path), len(val_path), len(test_path)))

    monitor.speak("Load data ...")
    if not args.val:
        monitor.speak("train mode")
        train_data = Data(train_path,
                          train=True,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))
    else:
        monitor.speak("val mode")
        train_data = Data(val_path,
                          train=True,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))

    test_data = Data(test_path,
                     train=True,
                     transforms=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))
                     ]))

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=multiprocessing.cpu_count(),
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=multiprocessing.cpu_count(),
                             pin_memory=True)

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()

    learning_rate = args.lr
    if args.adam:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=save_path, comment=args.comment)

    iter_idx = 0
    n_iter = args.n_iter
    val_interval = args.val_interval
    print_interval = 10

    lr = learning_rate

    best_test_loss = 4
    net.train()
    acc = 0
    acc_denominator = 0
    while True:
        for batch_idx, sample in enumerate(train_loader):
            img = sample['img'].to(device)
            feature = sample['feature'].to(device)
            target = sample['target'].to(device)
            iter_idx += 1
            lr = adjust_learning_rate(optimizer, iter_idx, n_iter, init_lr=learning_rate)
            output = net(img, feature)
            loss = criterion(output, target)
            pred_label = torch.argmax(output, dim=1)
            acc += torch.sum(pred_label == target)
            acc_denominator += img.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if iter_idx % print_interval == 0:
            #     monitor.speak('Iter: {}/{}\tLoss:{}\tLR: {}'.format(iter_idx, n_iter, "??", lr))
            if iter_idx % print_interval == 0:
                acc = acc / acc_denominator * 100
                monitor.speak(
                    'Iter: {}/{}\tLoss:{:.6f}\tACC: {:.2f}\tLR: {}'.format(iter_idx, n_iter, loss.item(), acc, lr))
                writer.add_scalar("train/train_acc", acc, iter_idx)
                acc = 0
                acc_denominator = 0
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                test_loss = 0.0
                net.eval()
                with torch.no_grad():
                    acc = 0.0
                    for batch_idx, sample in enumerate(test_loader):
                        img = sample['img'].to(device)
                        feature = sample['feature'].to(device)
                        target = sample['target'].to(device)
                        output = net(img, feature)
                        pred_label = torch.argmax(output, dim=1)
                        acc += torch.sum(pred_label == target).item()
                        loss = criterion(output, target)
                        test_loss += loss.item() * img.shape[0]
                    acc = acc / len(test_data)
                    test_loss = test_loss / len(test_data)

                    monitor.speak('Test Loss: {:.6f},acc:{:.4f}'.format(test_loss, acc))
                    writer.add_scalar("train/test_loss", test_loss, iter_idx)
                    writer.add_scalar("train/acc", acc, iter_idx)
                if test_loss < best_test_loss:
                    torch.save(net.state_dict(), save_path / "model{}".format(iter_idx))
                    monitor.speak("test loss: {:.6f} < best: {:.6f},save model".format(test_loss, best_test_loss))
                    best_test_loss = test_loss
                net.train()

        if iter_idx > n_iter:
            monitor.speak("Done")
            break
