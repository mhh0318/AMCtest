#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/1 22:30
@author: merci
"""
import time
import torch.optim as optim
from dataLoader import *
from model import *
from utils import *
from test import test
from train import train

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ResNet(BasicBlock,[2]*11).to(device)
    train_loader = data.DataLoader(dataset=Signals(train=True, unknown=2), batch_size=128,shuffle=True)
    test_loader = data.DataLoader(dataset=Signals(train=False, unknown=2), batch_size=128, shuffle=False)
    optimizer = optim.Adam(model.parameters())
    # Resume==0 : Train phase. Resume==1 : Test phase.
    resume = 0
    if resume == 0:
        accur = []
        for epoch in range(0, 300):
            start = time.time()
            train(train_loader, model, optimizer, device, epoch)
            end = time.time()
            print('Train time: {:.2f} s for one epoch'.format(end - start))
            start = time.time()
            prediction, probability, acc= test(test_loader, model, device)
            end = time.time()
            print('Test time: {:.2f} s'.format(end - start))
            accur.append(float(acc))

            conf = np.zeros([12,12])
            confnorm = np.zeros([12,12])
            labels = test_loader.dataset.y_data.tolist()
            for i in range(0, len(prediction)):
                j = labels[i]
                k = prediction[i]
                conf[j, k] = conf[j, k] + 1
            for i in range(12):
                confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
            plot_cm(confnorm[:, :])
            #plt.show()
            plt.savefig('/home/hu/AMCD3/Merge/R2048-Merge-FineLoss—0,1.png')
            torch.save(model, '/home/hu/AMCD3/Merge/Model-FineLoss-0,1.ckpt')
    elif resume == 1:
        model = torch.load('/home/hu/AMCD3/Merge/Model.ckpt').to(device)
        prediction, probability, acc = test(test_loader, model, device)
        conf = np.zeros([11, 11])
        confnorm = np.zeros([11, 11])
        labels = test_loader.dataset.y_data.tolist()
        for i in range(0, len(prediction)):
            j = labels[i]
            k = prediction[i]
            conf[j, k] = conf[j, k] + 1
        for i in range(11):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plot_cm(confnorm[:, :])
        plt.show()
    elif resume == 2:
        '''
        kk = np.load("/home/hu/AMCpri/Aug/2FSK_200Hz_30s_4kHz.npy")
        number = kk.size // 2048
        ks = np.array_split(kk, number)  #z or kk
        all = []
        for i in ks:
            signal = np.ones((2, 2048))
            for j in range(2048):
                signal[0, j] = np.real(i[j])
                signal[1, j] = np.imag(i[j])
            all.append(signal)
        trainX1 = np.array(all)
        '''
        import scipy.io as scio
        z = scio.loadmat("/home/hu/AMCpri/Aug/X_mi.mat")
        z = z['X_norm']
        trainX = z
        # trainX = (trainX - trainX.mean(axis=2, keepdims=True)) / trainX.std(axis=2, keepdims=True)

        z0 = scio.loadmat("/home/hu/AMCpri/Aug/X_res.mat")
        z0 = z0['X_res']
        realz = np.real(z0)
        imagz = np.imag(z0)
        trainX = np.stack((realz,imagz),axis=1)

        t0 = np.sqrt(np.mean(np.mean(np.square(trainX), axis=2), axis=0))
        trainX[:, 0, :] = trainX[:, 0, :] / t0[0]
        trainX[:, 1, :] = trainX[:, 1, :] / t0[1]

        model = torch.load('/home/hu/AMCD3/Merge/Model-Digital.ckpt').to(device)
        model.eval()
        correct = 0
        predictions = []
        probability = []

        dataa = torch.Tensor(trainX).to(device, dtype=torch.float)
        labels = torch.Tensor((np.ones(58)*4)).long().to(device)

        output = model(dataa)
        probability.extend(output[0].cpu().detach().numpy())
        pred = torch.max(output[0], 1)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum()
        predictions.extend(pred.cpu().numpy())
        acc = 100. * correct / 58
        print('\nTest set: Accuracy: {}\t'.format(acc))

if __name__ == "__main__":
    main()