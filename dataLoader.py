#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/1 22:16
@author: merci
"""
import os
import numpy as np
import h5py
import torch
from torch.utils import data


class Signals(data.Dataset):
    def __init__(self, train=True, unknown=0):
        self.train = train
        self.type = type
        if self.train == True:
            root = '/home/hu/AMCD3/Signals-2048/Train'
        else:
            root = '/home/hu/AMCD3/Signals-2048/Test_extralabel'
        files = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        x = []
        y = []
        # file_name = [f for f in os.listdir(root) if 'QAM' not in f and '8PSK' not in f]+['QAM','8PSK']
        file_name = [f for f in os.listdir(root) if 'SB' not in f]
        #labels = list(range(12))
        labels = list(range(14))
        zippedict = dict(zip(file_name, labels))
        for f in files:
            if unknown == 0:
                if 'MSK' in f or 'BPSK' in f:
                    continue
                else:
                    with h5py.File(f, 'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                        if 'QAM' in f:
                            y0 = np.full(y0.shape[0], zippedict['QAM'])
                        elif '8PSK' in f:
                            y0 = np.full(y0.shape[0], zippedict['8PSK'])
                        else:
                            y0 = np.full(y0.shape[0], zippedict[os.path.basename(f)])
                    x.append(x0)
                    y.append(y0)
                    print(f)
                    print(set(y0))
            elif unknown == 1:
                if 'SB' in f:
                    continue
                else:
                    with h5py.File(f, 'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                        if 'QAM' in f:
                            y0 = np.full(y0.shape[0], zippedict['QAM'])
                        elif '8PSK' in f:
                            y0 = np.full(y0.shape[0], zippedict['8PSK'])
                        else:
                            y0 = np.full(y0.shape[0], zippedict[os.path.basename(f)])
                    x.append(x0)
                    y.append(y0)
                    print(f)
                    print(y0[0])
            elif unknown == 2:
                if 'SB' in f:
                    continue
                else:
                    with h5py.File(f, 'r') as db:
                        x0 = db['Signal'][:][:]
                        y0 = db['Type'][:]
                        y0 = np.full(y0.shape[0], zippedict[os.path.basename(f)])
                        x.append(x0)
                        y.append(y0)
                        print(f)
                        print(y0[0])
        x = np.concatenate(x, axis=1)
        y = np.ravel(np.concatenate(y))
        self.x_data = torch.from_numpy(x)
        self.x_data = self.x_data.permute(1, 0, 2)
        self.y_data = torch.from_numpy(y)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
