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

    def __init__(self, img_paths, feat_path, train=True, transforms=None):
        self.img_paths = img_paths
        self.feat_path = feat_path
        self.train = train
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img_file_name = self.img_paths[idx].split('/')[-1]
        feat_file_name = img_file_name.split('.')[0] + '.npy'
        feature = np.load(os.path.join(self.feat_path, feat_file_name))
        feature = feature / np.sum(feature)

        if self.transforms is not None:
            img = self.transforms(img)

        sample = {}
        if self.train:
            target = int(self.img_paths[idx].split('/')[-1].split('_')[-1].split('.')[0]) - 1
            sample['img'] = img
            sample['feature'] = torch.from_numpy(feature).float()
            sample['target'] = target
        else:
            sample['img'] = img
            sample['feature'] = feature
        return sample


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


class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()
        self.feature_extracter = nn.Sequential(*list(resnet18().children())[:-1])
        self.fc = nn.Linear(512 + 24, num_classes)

    def forward(self, img, text):
        feature = self.feature_extracter(img)
        feature = feature.view((feature.shape[0], -1))
        out = torch.cat((feature, text), dim=1)
        out = self.fc(out)
        return out


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
    parser.add_argument('--val', action='store_true', default=False,
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
        monitor.speak("train mode")
        train_data = Data(splited_data.train,
                          feat_path='/userhome/bigdata/train/visit_feat',
                          train=True,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))
    else:
        monitor.speak("val mode")
        train_data = Data(splited_data.val,
                          train=True,
                          transforms=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))
                          ]))

    test_data = Data(splited_data.test,
                     feat_path='/userhome/bigdata/train/visit_feat',
                     train=True,
                     transforms=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))
                     ]))

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True)

    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()

    learning_rate = args.lr
    if args.adam:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=0.0001)

    writer = SummaryWriter()

    iter_idx = 0
    n_iter = 64000
    val_interval = 1000
    print_interval = 10

    lr = learning_rate

    best_test_loss = 4
    net.train()
    while True:
        for batch_idx, sample in enumerate(train_loader):
            img = sample['img'].to(device)
            feature = sample['feature'].to(device)
            target = sample['target'].to(device)
            iter_idx += 1
            lr = adjust_learning_rate(optimizer, iter_idx, init_lr=learning_rate)
            output = net(img, feature)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if iter_idx % print_interval == 0:
            #     monitor.speak('Iter: {}/{}\tLoss:{}\tLR: {}'.format(iter_idx, n_iter, "??", lr))
            if iter_idx % print_interval == 0:
                monitor.speak('Iter: {}/{}\tLoss:{:.6f}\tLR: {}'.format(iter_idx, n_iter, loss.item(), lr))
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
                    torch.save(net.state_dict(), r"./result/model{}".format(iter_idx))
                    monitor.speak("test loss: {:.6f} < best: {:.6f},save model".format(test_loss, best_test_loss))
                    best_test_loss = test_loss
                net.train()

        if iter_idx > n_iter:
            monitor.speak("Done")
            break
