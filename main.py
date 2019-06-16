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
from torchvision.models import resnet18, vgg19_bn
from pathlib import Path
import pandas as pd
from model import ResNet20, ResNetImg, ResNet20_64
import multiprocessing
from datetime import datetime
import time
from tqdm import tqdm


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

    def __init__(self, img_files, visit_path, monitor=None, train=True, val=False, only_visit=False, norm_visit=False,
                 transforms=None):
        self.img_files = img_files
        self.visit_path = visit_path
        self.train = train
        self.transforms = transforms
        self.val = val
        self.only_visit = only_visit
        self.norm_visit = norm_visit

        ms_path = Path(Path(visit_path).parent)
        self.mean = np.load(ms_path / 'visit_mean.npy')
        self.std = np.load(ms_path / 'visit_std.npy')

        self.std[np.where(self.std == 0)] = 1

        if self.val:  # about 10~20s accerleration for val(4000) each epoch, consumes 2 min
            if not monitor:
                ...
            else:
                monitor.speak("load all to memory")
            self.imgs = []
            self.visits = []
            for img_file in tqdm(self.img_files):
                img = np.array(Image.open(img_file))
                img_name = img_file.split('/')[-1]
                feature = np.load(self.visit_path + '/' + img_name.split('.')[0] + '.npy')
                self.imgs.append(img)
                self.visits.append(feature)
        else:
            if not monitor:
                ...
            else:
                monitor.speak("lazy load")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_name = str(img_file).split('/')[-1]
        if not self.only_visit:
            if self.val:
                img = self.imgs[idx]
                feature = self.visits[idx]
            else:
                img = Image.open(img_file)
                feature = np.load(self.visit_path + '/' + img_name.split('.')[0] + '.npy')
            if self.transforms is not None:
                img = self.transforms(img)
        else:
            if self.val:
                feature = self.visits[idx]
            else:
                feature = np.load(self.visit_path + '/' + img_name.split('.')[0] + '.npy')
            img = 0
        if self.norm_visit:
            feature = (feature - self.mean) / self.std
        sample = {}
        if self.train:
            target = int(img_name.split('.')[0].split('_')[-1]) - 1
            sample['img'] = img
            sample['feature'] = feature / 1.0
            sample['target'] = target
        else:
            sample['img'] = img
            sample['feature'] = feature / 1.0
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


class FC(nn.Module):
    def __init__(self, *args):
        super(FC, self).__init__()
        # self.fc1 = nn.Linear(4368, 1000)
        # self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(4368, 9)

    def forward(self, img, x):
        x = x.view(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net3(nn.Module):
    def __init__(self, resnet20=False, only_visit=False, num_classes=9):
        super(Net3, self).__init__()
        self.visit_feature_extracter1 = ResNet20(in_channel=7)
        self.visit_feature_extracter2 = ResNet20(in_channel=24)
        self.visit_feature_extracter3 = ResNet20(in_channel=26)
        self.resnet20 = resnet20
        self.only_visit = only_visit
        if not self.only_visit:
            if self.resnet20:
                self.img_feature_extracter = ResNetImg(3)
                self.fc = nn.Linear(128 + 64, num_classes)
            else:
                self.img_feature_extracter = nn.Sequential(*list(resnet18(pretrained=False).children())[:-1])
                # self.img_feature_extracter = nn.Sequential(*list(vgg19_bn(pretrained=True).children())[:-1])
                self.fc = nn.Linear(512 + 128, num_classes)
                # self.fc = nn.Linear(25152, num_classes)
        else:
            self.dropout = nn.Dropout()
            self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, img, text):
        if not self.only_visit:
            img_feature = self.img_feature_extracter(img).view((img.size(0), -1))
            visit_feature = self.visit_feature_extracter(text)
            # print(img_feature.shape)
            out = torch.cat((img_feature, visit_feature), dim=1)
        else:
            out1 = self.visit_feature_extracter1(text)
            out2 = self.visit_feature_extracter2(text.permute(0, 3, 1, 2))
            out3 = self.visit_feature_extracter3(text.permute(0, 2, 1, 3))
        out = self.dropout(self.fc(torch.cat((out1, out2, out3), dim=1)))
        return out


class Net3R(nn.Module):
    def __init__(self, resnet20=False, only_visit=False, dropout=False, num_classes=9):
        super(Net3R, self).__init__()
        self.visit_feature_extracter1 = ResNet20(in_channel=1)
        self.visit_feature_extracter2 = ResNet20(in_channel=1)
        self.visit_feature_extracter3 = ResNet20(in_channel=1)
        self.resnet20 = resnet20
        self.only_visit = only_visit
        self.dropout = dropout
        if not self.only_visit:
            if self.resnet20:
                self.img_feature_extracter = ResNetImg(3)
                self.fc = nn.Linear(128 + 64, num_classes)
            else:
                self.img_feature_extracter = nn.Sequential(*list(resnet18(pretrained=False).children())[:-1])
                # self.img_feature_extracter = nn.Sequential(*list(vgg19_bn(pretrained=True).children())[:-1])
                self.fc = nn.Linear(512 + 128 * 3, num_classes)
                if self.dropout:
                    self.dropout = nn.Dropout()
                # self.fc = nn.Linear(25152, num_classes)
        else:
            if self.dropout:
                self.dropout = nn.Dropout()
            self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, img, text):
        if not self.only_visit:
            img_feature = self.img_feature_extracter(img).view((img.size(0), -1))
            out1 = self.visit_feature_extracter1(text.sum(dim=1).unsqueeze(1))
            out2 = self.visit_feature_extracter2(text.sum(dim=2).unsqueeze(1))
            out3 = self.visit_feature_extracter3(text.sum(dim=3).unsqueeze(1))
            if self.dropout:
                out = self.dropout(self.fc(torch.cat((out1, out2, out3, img_feature), dim=1)))
            else:
                out = self.fc(torch.cat((out1, out2, out3, img_feature), dim=1))
        else:
            out1 = self.visit_feature_extracter1(text.sum(dim=1).unsqueeze(1))
            out2 = self.visit_feature_extracter2(text.sum(dim=2).unsqueeze(1))
            out3 = self.visit_feature_extracter3(text.sum(dim=3).unsqueeze(1))
            if self.dropout:
                out = self.dropout(self.fc(torch.cat((out1, out2, out3), dim=1)))
            else:
                out = self.fc(torch.cat((out1, out2, out3), dim=1))
        return out


class Net(nn.Module):
    def __init__(self, resnet20=False, only_visit=False, visit64=False, num_classes=9):
        super(Net, self).__init__()
        if not visit64:
            self.visit_feature_extracter = ResNet20(in_channel=7)
        else:
            self.visit_feature_extracter = ResNet20_64(in_channel=7)
        self.visit_dim = 64 if visit64 else 128
        self.resnet20 = resnet20
        self.only_visit = only_visit

        if not self.only_visit:
            if self.resnet20:
                self.img_feature_extracter = ResNetImg(3)
                self.fc = nn.Linear(128 + self.visit_dim, num_classes)
            else:
                self.img_feature_extracter = nn.Sequential(*list(resnet18(pretrained=False).children())[:-1])
                # self.img_feature_extracter = nn.Sequential(*list(vgg19_bn(pretrained=True).children())[:-1])
                self.fc = nn.Linear(512 + self.visit_dim, num_classes)
                # self.fc = nn.Linear(25152, num_classes)
        else:
            self.fc = nn.Linear(self.visit_dim, num_classes)

    def forward(self, img, text):
        if not self.only_visit:
            img_feature = self.img_feature_extracter(img).view((img.size(0), -1))
            visit_feature = self.visit_feature_extracter(text)
            # print(img_feature.shape)
            out = torch.cat((img_feature, visit_feature), dim=1)
        else:
            out = self.visit_feature_extracter(text)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Em blah blah blah')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training and testing (default: 64)')
    parser.add_argument('--n-iter', type=int, default=64000,
                        help='n-iter (default: 64000)')
    parser.add_argument('--val-interval', type=int, default=500,
                        help='val interveal (default: 500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use adam')
    parser.add_argument('--resnet20img', action='store_true', default=False,
                        help='use resnet20 for img feature')
    parser.add_argument('--stop', type=int, default=20,
                        help='stop when acc not increase')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--save-path', type=str, default='./test',
                        help='save path (default: ./test)')
    parser.add_argument('--val', action='store_true', default=False,
                        help='validation mode')
    parser.add_argument('--only-visit', action='store_true', default=False,
                        help='only use visit data')
    parser.add_argument('--fc', action='store_true', default=False,
                        help='use fc for visit')
    parser.add_argument('--norm-visit', action='store_true', default=False,
                        help='norm visit')
    parser.add_argument('--net3', action='store_true', default=False,
                        help='3 visit feat')
    parser.add_argument('--net3R', action='store_true', default=False,
                        help='3R visit feat')
    parser.add_argument('--dropout', action='store_true', default=False,
                        help='use dropout')
    parser.add_argument('--split91', action='store_true', default=False,
                        help='split train test = 9:1 (default: false)')
    parser.add_argument('--visit64', action='store_true', default=False,
                        help='use 64-d visit feat')
    parser.add_argument('--comment', type=str, default=str(datetime.today()),
                        help='comment')
    args = parser.parse_args()
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    event_path = os.path.join(save_path, args.comment)
    if not os.path.exists(event_path):
        os.mkdir(event_path)

    monitor = OutPutUtil(True, True, event_path)
    monitor.speak(args)
    batch_size = args.batch_size

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    monitor.speak("Split Data ...")
    if args.val:
        spliter = DataSpliter('/userhome/bigdata/train')
    else:
        if not args.split91:
            spliter = DataSpliter('/userhome/bigdata/train', ratio=[0.70, 0, 0.30])
        else:
            spliter = DataSpliter('/userhome/bigdata/train', ratio=[0.90, 0, 0.10])
    train_path = spliter.train
    val_path = spliter.val
    test_path = spliter.test

    visit_dir = '/userhome/bigdata/train/visit_feat'

    monitor.speak("OK\ntrain:{},val:{},test:{}".format(len(train_path), len(val_path), len(test_path)))

    monitor.speak("Load data ...")
    if not args.val:
        monitor.speak("train mode")
        train_data = Data(train_path,
                          visit_dir,
                          train=True,
                          val=False,
                          only_visit=args.only_visit,
                          transforms=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomCrop((88, 88)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]), monitor=monitor, norm_visit=args.norm_visit)
    else:
        monitor.speak("val mode")
        train_data = Data(val_path,
                          visit_dir,
                          train=True,
                          val=False,
                          only_visit=args.only_visit,
                          transforms=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomCrop((88,88)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]), monitor=monitor, norm_visit=args.norm_visit)

    test_data = Data(test_path,
                     visit_dir,
                     train=True,
                     transforms=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))
                     ]), monitor=monitor)

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
    if args.fc:
        net = FC().to(device)
    elif args.net3:
        net = Net3(only_visit=args.only_visit).to(device)
    elif args.net3R:
        net = Net3R(only_visit=args.only_visit, dropout=args.dropout).to(device)
    else:
        net = Net(args.resnet20img, only_visit=args.only_visit, visit64=args.visit64).to(device)

    monitor.speak(net)
    criterion = nn.CrossEntropyLoss()

    learning_rate = args.lr
    if args.adam:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=event_path, comment=args.comment)

    iter_idx = 0
    n_iter = args.n_iter
    val_interval = args.val_interval
    print_interval = 10

    lr = learning_rate

    best_acc = 0
    stop_flag = 0
    best_test_loss = 4
    net.train()
    acc = 0
    acc_dis = 0
    acc_denominator = 0
    distribution = torch.Tensor([0.2386, 0.1884, 0.0897, 0.0340, 0.0866, 0.1377, 0.0879, 0.0654, 0.0717]).to(device)
    torch.cuda.synchronize()
    start = time.time()
    while True:
        for batch_idx, sample in enumerate(train_loader):
            img = sample['img'].to(device)
            feature = sample['feature'].float().to(device)
            target = sample['target'].to(device)
            iter_idx += 1
            lr = adjust_learning_rate(optimizer, iter_idx, n_iter, init_lr=learning_rate)
            output = net(img, feature)
            loss = criterion(output, target)
            pred_label = torch.argmax(output, dim=1)
            acc += torch.sum(pred_label == target).item()
            acc_denominator += img.size(0)

            output = torch.nn.functional.softmax(output, dim=1)
            output = output / distribution
            pred_label_dis = torch.argmax(output, dim=1)
            acc_dis += torch.sum(pred_label_dis == target).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if iter_idx % print_interval == 0:
            #     monitor.speak('Iter: {}/{}\tLoss:{}\tLR: {}'.format(iter_idx, n_iter, "??", lr))
            if iter_idx % print_interval == 0:
                acc = acc / acc_denominator * 100
                acc_dis = acc_dis / acc_denominator * 100
                torch.cuda.synchronize()
                end = time.time()
                message = 'Iter: {}/{}\tLoss:{:.6f}\tACC: {:.2f}\tACC_d: {:.2f}\ttime/print:{:.4f}\tLR: {}'
                monitor.speak(message.format(iter_idx, n_iter,
                                             loss.item(), acc,
                                             acc_dis,
                                             end - start,
                                             lr))
                writer.add_scalar("train/train_acc", acc, iter_idx)
                writer.add_scalar("train/train_acc_dis", acc_dis, iter_idx)
                acc = 0
                acc_dis = 0
                acc_denominator = 0
                torch.cuda.synchronize()
                start = time.time()
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                test_loss = 0.0
                net.eval()
                try:
                    with torch.no_grad():
                        acc = 0.0
                        acc_dis = 0.0
                        for batch_idx, sample in tqdm(enumerate(test_loader)):
                            img = sample['img'].to(device)
                            feature = sample['feature'].float().to(device)
                            target = sample['target'].to(device)
                            output = net(img, feature)
                            pred_label = torch.argmax(output, dim=1)
                            acc += torch.sum(pred_label == target).item()

                            # consider distribution
                            output = torch.nn.functional.softmax(output, dim=1)
                            output = output / distribution
                            pred_label_dis = torch.argmax(output, dim=1)
                            acc_dis += torch.sum(pred_label_dis == target).item()

                            loss = criterion(output, target)
                            test_loss += loss.item() * img.shape[0]
                        acc = acc / len(test_data)
                        acc_dis = acc_dis / len(test_data)
                        test_loss = test_loss / len(test_data)

                        monitor.speak('Test Loss: {:.6f},acc:{:.4f},acc d{:.4f}'.format(test_loss, acc, acc_dis))
                        writer.add_scalar("train/test_loss", test_loss, iter_idx)
                        writer.add_scalar("train/acc", acc, iter_idx)
                        writer.add_scalar("train/acc_d", acc_dis, iter_idx)
                    if test_loss < best_test_loss and not args.val:
                        torch.save(net.state_dict(), event_path + '/' + "model{}".format(iter_idx))
                        monitor.speak("test loss: {:.6f} < best: {:.6f},save model".format(test_loss, best_test_loss))
                        best_test_loss = test_loss
                    if acc > best_acc:
                        torch.save(net.state_dict(), event_path + '/' + "model{}".format(iter_idx))
                        monitor.speak("test loss: {:.6f} < best: {:.6f},save model".format(test_loss, best_test_loss))
                        best_acc = acc
                        stop_flag = 0
                    else:
                        stop_flag += 1
                        monitor.speak('stop flag: {}'.format(stop_flag))
                    if stop_flag >= args.stop:
                        break
                except KeyboardInterrupt:
                    monitor.speak("stop eval")
                    pass
                net.train()

        if iter_idx > n_iter or stop_flag >= args.stop:
            monitor.speak("Done")
            break
