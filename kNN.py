#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2019/9/27 15:15
@author: merci
"""
import sklearn
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
import h5py
import numpy as np
import torch
from torch.utils import data
from ResAMC import HDF5Data, test, device, plot_cm, ResNet, conv1x1, conv3x3, BasicBlock


root = "/home/hu/AMCpri/data/test-new.h5"
db = h5py.File(root,'r')
x = db['Signal'][:]
y_raw = db['Type'][:]

classes = 14
filter_signals = []


test_loader = data.DataLoader(dataset=HDF5Data(train=False), batch_size=128, shuffle=False)

model = torch.load('/home/hu/AMCpri/CONF/Joint++.ckpt')
prediction, p_final, acc = test(test_loader, model)
model.eval()
code = []

for signal, labels in test_loader:
    signal = signal.to(device, dtype=torch.float)
    labels = labels.long().to(device)
    output = model(signal)
    code.append(output[1].detach().cpu().numpy().squeeze())
code = np.concatenate((np.array(code[:-1]).reshape(-1,32,32),code[-1]),0)
nsamples, nx, ny = code.shape
d2_train_dataset = code.reshape((nsamples,nx*ny))


np.random.seed(0)
'''
# x = (x - x.mean(axis=2, keepdims=True)) / x.std(axis=2, keepdims=True)
# norm_sig = np.squeeze(np.linalg.norm([x[0],x[1]],axis=0,keepdims=True))

inds = {}

for i in range(classes):
    indecies = np.where(y_raw == i)[0]
    for index in indecies:
        inds[index] = i
    h = norm_sig[indecies,:]
    filter_signals.append(h[:4000])
'''
result = []
distance = []

for i in range(classes):
    neigh = NearestNeighbors(n_neighbors=10,n_jobs=-1)
    # neigh.fit(norm_signal)
    neigh.fit(d2_train_dataset)
    A = neigh.kneighbors([d2_train_dataset[i]])
    distance.append(A[0])
    result.append(A[1])

r_ = []

for i in range(classes):
    r_ += result[i].tolist()
r_ = np.array(r_).flatten().tolist()
'''
rr = np.array(list(map(lambda x: inds[x],r_))).reshape(14,-1,10)

r0 = rr.reshape(14,-1)

types = []
counts = []
for item in r0:
    tp,cnts = np.unique(item,return_counts=True)
    types.append(tp)
    counts.append(cnts)
'''
print('ok')