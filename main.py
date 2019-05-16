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


class DataSpliter:
    sub_paths = ['001', '002', '003', '004', '005', '006', '007', '008', '009']

    def __init__(self, path, ratio=[0.8, 0.1, 0.1]):
        self.path = path
        self.ratio = ratio
        self.train = []
        self.val = []
        self.test = []
        for sub_path in self.sub_paths:
            class_path = os.path.join(path, sub_path)
            filenames = os.listdir(class_path)
            filenames = [os.path.join(class_path, filename) for filename in filenames]
            num = len(filenames)
            shuffle(filenames)
            self.train += filenames[0:int(ratio[0] * num)]
            self.val += filenames[int(ratio[0] * num):int((ratio[0] + ratio[1]) * num)]
            self.test += filenames[int((ratio[0] + ratio[1]) * num):]


class Data(Dataset):

    def __init__(self, filepaths, train=True, transforms=None, target_transforms=None):
        self.filepaths = filepaths
        self.train = train
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx])

        if self.train:
            target = int(self.filepaths[idx].split('/')[-1].split('_')[-1].split('.')[0]) - 1

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target if self.train else img


def adjust_learning_rate(optimizer, iteration, init_lr=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if iteration <= 32000:
        lr = init_lr
    elif 32000 < iteration <= 48000:
        lr = init_lr / 10
    else:
        lr = init_lr / 100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Em blah blah blah')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training and testing (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-path', type=str, default='./train.log',
                        help='path to save log file (default: ./train.log)')
    parser.add_argument('--adam', action='store_true', default=False,
                        help='use adam')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0,
                        help='momentum (default: 0)')
    parser.add_argument('--save-path', type=str, default='./result',
                        help='save path (default: ./result)')
    parser.add_argument('--val',action='store_true',default=False,
                        help='validation mode')
    args = parser.parse_args()

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    monitor = OutPutUtil(True, True, args.log_path)

    batch_size = args.batch_size

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    splited_data = DataSpliter('/userhome/bigdata/train')

    if not args.val:
        train_data = Data(splited_data.train,
                          train=True,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))
    else:
        train_data = Data(splited_data.val,
                        train=True,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))

    test_data = Data(splited_data.test,
                     train=True,
                     transforms=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))
                     ]))

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=1)

    net = resnet18(**{'num_classes': 9}).to(device)

    criterion = nn.CrossEntropyLoss()

    learning_rate = args.lr
    if args.adam:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter()

    iter_idx = 0
    n_iter = 64000
    val_interval = 1000
    print_interval = 10

    lr = learning_rate

    best_test_loss = 4
    while True:
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            iter_idx += 1
            lr = adjust_learning_rate(optimizer, iter_idx, init_lr=learning_rate)
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_idx % print_interval == 0:
                monitor.speak('Iter: {}/{}\tLoss:{:.6f}\tLR: {}'.format(iter_idx, n_iter, loss.item(), lr))
            writer.add_scalar("train/train_loss", loss.item(), iter_idx)

            if iter_idx % val_interval == 0:
                test_loss = 0.0
                with torch.no_grad():
                    acc = 0.0
                    for data, target in test_loader:
                        data = data.to(device)
                        target = target.to(device)
                        output = net(data)
                        pred_label = torch.argmax(output, dim=1)
                        acc += torch.sum(pred_label == target).item()
                        loss = criterion(output, target)
                        test_loss += loss.item() * data.shape[0]
                    acc = acc / len(test_data)
                    test_loss = test_loss / len(test_data)

                    monitor.speak('Test Loss: {:.6f},acc:{:.4f}'.format(test_loss, acc))
                    writer.add_scalar("train/test_loss", test_loss, iter_idx)
                    writer.add_scalar("train/acc", acc, iter_idx)
                if test_loss < best_test_loss:
                    torch.save(net.state_dict(), r"./result/model{}".format(iter_idx))
                    monitor.speak("test loss: {:.6f} < best: {:.6f},save model".format(test_loss, best_test_loss))
                    best_test_loss = test_loss

        if iter_idx > n_iter:
            monitor.speak("Done")
            break
