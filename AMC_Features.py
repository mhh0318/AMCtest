#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/11/27 13:28
@author: merci
"""
import os
import numpy as np
from sklearn import neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import time
from ResModel import *


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
np.seterr(divide='ignore', invalid='ignore')

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1): #3 and 5
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        # out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 2
        self.planes = 32
        self.bn1 = norm_layer(self.inplanes)
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer1 = self._make_layers(block, 32, layers[0])
        self.layer2 = self._make_layers(block, 32, layers[1])
        self.layer3 = self._make_layers(block, 32, layers[2])
        self.layer4 = self._make_layers(block, 32, layers[3])
        self.layer5 = self._make_layers(block, 32, layers[4])
        self.layer6 = self._make_layers(block, 32, layers[5])
        self.layer7 = self._make_layers(block, 32, layers[6])
        self.layer8 = self._make_layers(block, 32, layers[7])
        self.layer9 = self._make_layers(block, 32, layers[8])
        self.layer10 = self._make_layers(block, 32, layers[9])
        self.layer11 = self._make_layers(block, 32, layers[10])
        '''
        self.layer11 = self._make_layers(block, 32, layers[10])
        self.layer12 = self._make_layers(block, 32, layers[11])
        self.layer13 = self._make_layers(block, 32, layers[12])
        self.layer14 = self._make_layers(block, 32, layers[13])
        self.layer15 = self._make_layers(block, 32, layers[14])
        self.layer16 = self._make_layers(block, 32, layers[15])
        self.layer17 = self._make_layers(block, 32, layers[16])
        self.layer18 = self._make_layers(block, 32, layers[17])
        self.layer19 = self._make_layers(block, 32, layers[18])
        self.layer20 = self._make_layers(block, 32, layers[19])
        '''

        self.clf = nn.Sequential(
            nn.Linear(256, 128),
            nn.SELU(True),
            nn.AlphaDropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.AlphaDropout(0.5),
            nn.Linear(128, 3)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layers(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(conv1x1(self.inplanes, self.planes))
        self.inplanes = 32
        layers.append(block(self.planes, planes, stride))
        layers.append(block(self.planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.mp(self.layer1(x))
        x = self.mp(self.layer2(x))
        x = self.mp(self.layer3(x))
        x = self.mp(self.layer4(x))
        x = self.mp(self.layer5(x))
        x = self.mp(self.layer6(x))
        x = self.mp(self.layer7(x))
        x = self.mp(self.layer8(x))
        x = self.mp(self.layer9(x))
        x = self.mp(self.layer10(x))
        x = self.mp(self.layer11(x))

        x = x.view(x.size(0), -1)

        code = x

        x = self.clf(x)

        return x, code


class SimuData(data.Dataset):
    def __init__(self, train=True, type=None):
        self.train = train
        self.type = type
        if self.train:
            root = "/home/hu/NewSignal/train.h5"
            #root = '/home/hu/AMCD3/Signals-2048/QAMs-train2.h5'
        else:
            root = "/home/hu/NewSignal/test.h5"
        db = h5py.File(root,'r')
        x = db['Signal'][:][:]
        y = db['Type'][:]
        if self.type:
            mask = np.isin(y, self.type)
            x = x[:,mask,:]
            y = y[mask]
        lable_list = np.unique(y)
        for i in lable_list:
            y[y==i] = np.squeeze(np.argwhere(lable_list==i))
        # x = (x-x.mean(axis=2, keepdims=True))/x.std(axis=2,keepdims=True)
        self.label = torch.from_numpy(y)
        self.x_data = torch.from_numpy(x)
        self.x_data = self.x_data.permute(1, 0, 2) #(n,2,1024)
        self.y_data = torch.from_numpy(y)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class LongerSequence(data.Dataset):
    def __init__(self, train=True, unknown=0):# 0:all 14 2:11trained 1:only unknown 3:only QAMs 4:Double QAMs
        self.train = train
        self.type = type
        if self.train:
            root = '/home/hu/AMCD3/Signals-2048/Train'
        else:
            root = '/home/hu/AMCD3/Signals-2048/Test'
        files = [os.path.join(root,f) for f in os.listdir(root) if os.path.isfile(os.path.join(root,f))]
        x = []
        y = []
        for f in files:
            if unknown == 2:
                if 'SSB' in f or 'D8PSK' in f or '2FSK' in f:
                    continue
                else:
                    with h5py.File(f,'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                    x.append(x0)
                    y.append(y0)
                    print(f)
            elif unknown == 1:
                if 'SSB' in f or 'D8PSK' in f or '2FSK' in f:
                    with h5py.File(f, 'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                    x.append(x0)
                    y.append(y0)
            elif unknown == 0:
                with h5py.File(f,'r') as db:
                    x0 = db['Signal'][:][:]
                    y0 = db['Type'][:]
                x.append(x0)
                y.append(y0)
                print(f)
            elif unknown == 3:
                if 'QAM' in f:
                    with h5py.File(f, 'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                    x.append(x0)
                    y.append(y0)
            elif unknown == 4:
                if 'SB' in f or 'D8PSK' in f:
                    continue
                else:
                    with h5py.File(f,'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                    x.append(x0)
                    y.append(y0)
                    if 'QAM' in f:
                        x.append(x0)
                        y.append(y0)
        x = np.concatenate(x, axis=1)
        y = np.ravel(np.concatenate(y))
        self.label = torch.from_numpy(y)
        lable_list = np.unique(y)
        for i in lable_list:
            y[y == i] = np.squeeze(np.argwhere(lable_list == i))
        # x = (x-x.mean(axis=2, keepdims=True))/x.std(axis=2,keepdims=True)
        self.x_data = torch.from_numpy(x)
        self.x_data = self.x_data.permute(1, 0, 2)
        self.y_data = torch.from_numpy(y)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



def train(train_loader, model, optimizer, epoch=0, weight=None):
    model.train()
    train_losses = []
    # for epoch in range(epochs, 200):  # epochs number
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device, dtype=torch.float)  # [batch_size, 1, 2, 512]
        labels = labels.long().to(device)  # [batch_size]
        # labels = torch.max(labels, 1)[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs[0], labels)
        #loss = F.cross_entropy(outputs , labels)
        if weight is not None:
            weight = torch.tensor(weight).float().to(device)
            #loss =F.cross_entropy(outputs[0], labels,weight=weight)
            loss =F.cross_entropy(outputs, labels,weight=weight)
        # loss = F.nll_loss(outputs, labels)
        loss.backward()
        train_losses.append(loss.item())
        # print(outputs,loss)
        optimizer.step()
        if i % 1000 == 0:
            # print(outputs)
            print('Train Epoch :{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, i * len(data), len(train_loader.dataset), 100. * i / len(train_loader), loss.item()))
    return np.array(train_losses).mean()


def test(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    probability = []
    for data, labels in test_loader:
        data = data.to(device, dtype=torch.float)
        labels = labels.long().to(device)
        # labels = torch.max(labels, 1)[1].to(device)
        output = model(data)
        probability.extend(output[0].cpu().detach().numpy())
        #prob = F.softmax(output[0], dim=1)
        #prob = F.softmax(output, dim=1)
        # probability.extend(prob.cpu().detach().numpy())
        test_loss += F.cross_entropy(output[0],labels).item()
        #test_loss += F.cross_entropy(output,labels).item()
        # test_loss += F.nll_loss(output, labels).item()
        pred = torch.max(output[0], 1)[1]
        #pred = torch.max(output, 1)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum()
        predictions.extend(pred.cpu().numpy())
    acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\t'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return predictions, probability, acc


def plot_cm(cm, title='Confusion matrix', cmap=plt.cm.Reds,  labels=[]):
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('scaled')
    if labels is not None:
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, fontsize=20)
        plt.yticks(tick_marks, labels, fontsize=20)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    textstr = np.around(cm.dot(100), decimals=2).astype(str)
    for i in range(0, cm.shape[0]):
        for j in range(0, cm.shape[1]):
            if cm[i, j] > 0.5:
                ijcolor = 'white'
            else:
                ijcolor = 'black'
            plt.text(j - 0.3, i, textstr[i, j] + '%', size='xx-large' ,color=ijcolor)

def catlist(l):
    tmp = l.pop()
    l = np.array(l).reshape(-1,tmp.shape[1],tmp.shape[2])
    tmp = np.concatenate((l,tmp),axis=0)
    return tmp

def main():
    model = ResNet(BasicBlock,[2,2,2,2,2,2,2,2,2,2,2]).to(device)
    #train_loader = data.DataLoader(dataset=SimuData(train=True), batch_size=128,shuffle=True)
    #test_loader = data.DataLoader(dataset=SimuData(train=False), batch_size=128, shuffle=False)
    #train_loader = data.DataLoader(dataset=LongerSequence(train=True), batch_size=128, shuffle=True)
    #test_loader = data.DataLoader(dataset=LongerSequence(train=False), batch_size=128, shuffle=False)
    # 1: Train Nonlinear classifier  2: Train Linear classifier  3: Test
    resume = 2
    if resume == 1:
        train_loader = data.DataLoader(dataset=LongerSequence(train=True, unknown=2), batch_size=128, shuffle=True)
        test_loader = data.DataLoader(dataset=LongerSequence(train=False, unknown=2), batch_size=128, shuffle=False)
        print('NonLinear')
        optimizer = optim.Adam(model.parameters())
        accur = []
        for epoch in range(0, 100):
            torch.cuda.synchronize()
            start = time.time()
            train(train_loader, model, optimizer, epoch)
            torch.cuda.synchronize()
            end = time.time()
            print('Train time: {:.2f} s for one epoch'.format(end - start))
            torch.cuda.synchronize()
            start = time.time()
            prediction, probability, acc= test(test_loader, model)
            torch.cuda.synchronize()
            end = time.time()
            print('Test time: {:.2f} s'.format(end - start))
            accur.append(float(acc))

            conf = np.zeros([9,9])
            confnorm = np.zeros([9,9])
            labels = test_loader.dataset.y_data.tolist()
            for i in range(0, len(prediction)):
                j = labels[i]
                k = prediction[i]
                conf[j, k] = conf[j, k] + 1
            for i in range(9):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plot_cm(confnorm[:, :])
                    #labels=['BPSK', 'MSK', 'QPSK', '8PSK', '16QAM', '2FSK', '4FSK', '32QAM', '64QAM'])
            plt.savefig('/home/hu/AMCD3/Linear/R2048-except2fsk&ssb-{}.png'.format(epoch))
            torch.save(model, '/home/hu/AMCD3/Specialize/2048-new11.ckpt')
    elif resume == 2:
        train_loader = data.DataLoader(dataset=LongerSequence(train=True, unknown=2), batch_size=128, shuffle=True)
        test_loader = data.DataLoader(dataset=LongerSequence(train=False, unknown=2), batch_size=128, shuffle=False)

        import copy
        print('Linear')
        model_linear = copy.deepcopy(model)
        model_linear.clf = nn.Linear(32, 11)
        print(model_linear)

        # model_linear = torch.load('/home/hu/AMCD3/Linear/2048-11').to(device)
        optimizer = optim.Adam(model_linear.parameters())
        accur = []
        model_linear = model_linear.to(device)
        for epoch in range(0, 300):
            torch.cuda.synchronize()
            start = time.time()
            train(train_loader, model_linear, optimizer, epoch)
            torch.cuda.synchronize()
            end = time.time()
            print('Train time: {:.2f} s for one epoch'.format(end - start))
            torch.cuda.synchronize()
            start = time.time()
            prediction, probability, acc = test(test_loader, model_linear)
            torch.cuda.synchronize()
            end = time.time()
            print('Test time: {:.2f} s'.format(end - start))
            accur.append(float(acc))
            conf = np.zeros([11,11])
            confnorm = np.zeros([11,11])
            labels = test_loader.dataset.y_data.tolist()
            for i in range(0, len(prediction)):
                j = labels[i]
                k = prediction[i]
                conf[j, k] = conf[j, k] + 1
            for i in range(11):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plot_cm(confnorm[:, :],)
                    #labels=['BPSK', 'MSK', 'QPSK', 'DQPSK' ,'8PSK', '16QAM', '2FSK', '4FSK', '8FSK', '32QAM', '64QAM'])
            plt.savefig('/home/hu/AMCD3/Linear/R2048-except2fsk&ssb-{}.png'.format(epoch))
            torch.save(model_linear, '/home/hu/AMCD3/Linear/2048-new11.ckpt')

        '''
        model = torch.load('/home/hu/AMCD3/Linear/2048-11').to(device)
        prediction, probability, acc = test(test_loader, model)

        L = test_loader.dataset.y_data
        p = np.array(probability)
        pp = []
        cmap = plt.get_cmap("Set3", 11)
        #for i in range(11):
        for i in [5,9,10]:
                tp_ = np.where(L == i)
                p_ = np.squeeze(p[tp_, :])
                pp.append(p_)
        for i in pp:
            fig = plt.figure()
            ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
            ax_cb = fig.add_axes([0.05, 0.05, 0.9, 0.1])
            for j in range(11):
                ax.hist(i[:,j],'auto',alpha=0.8,color=cmap(j))
                cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,
                                                orientation='horizontal')
                ticks = []
                cb1.set_ticks(ticks, update_ticks=True)
            plt.show()
        '''
        '''
        for i in pp:
            for j in range(11):
                plt.subplot(3,4,i)
                plt.hist(i[:, j], 'auto', color=cmap(j))
                plt.show()
        '''
        '''
        conf = np.zeros([11, 11])
        confnorm = np.zeros([11, 11])
        labels = test_loader.dataset.y_data.tolist()
        for i in range(0, len(prediction)):
            j = labels[i]
            k = prediction[i]
            conf[j, k] = conf[j, k] + 1
        for i in range(11):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plot_cm(confnorm[:, :], )
        plt.savefig('/home/hu/R2048-11.png')
        '''
    elif resume == 3:
        model = torch.load('/home/hu/AMCD3/Specialize/2048.ckpt').to(device)
        test_loader = data.DataLoader(dataset=SimuData(train=False), batch_size=128, shuffle=False)
        prediction, probability, acc = test(test_loader, model)
        conf = np.zeros([3,3])
        confnorm = np.zeros([3,3])
        labels = test_loader.dataset.y_data.tolist()
        for i in range(0, len(prediction)):
            j = labels[i]
            k = prediction[i]
            conf[j, k] = conf[j, k] + 1
        for i in range(3):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plot_cm(confnorm[:, :])
        plt.savefig('/home/hu/AMCD3/Specialize/2048.png')
    elif resume == 4: #specialize train
        # model = wide_resnet50_2(pretrained=False, progress=True).to(device)
        train_loader = data.DataLoader(dataset=SimuData(train=True), batch_size=128, shuffle=True)
        test_loader = data.DataLoader(dataset=SimuData(train=False), batch_size=128, shuffle=False)
        optimizer = optim.Adam(model.parameters())
        for i in range(200):
            torch.cuda.synchronize(device)
            start = time.time()
            train(train_loader, model, optimizer, i)
            torch.cuda.synchronize(device)
            end = time.time()
            print('Train time: {:.2f} s for one epoch'.format(end - start))
            torch.cuda.synchronize(device)
            start = time.time()
            test(test_loader, model)
            torch.cuda.synchronize(device)
            end = time.time()
            print('Test time: {:.2f} s'.format(end - start))
            torch.save(model, '/home/hu/AMCD3/Specialize/2048.ckpt')





if __name__ == "__main__":
    main()