#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/1 22:23
@author: merci
"""
import torch
import torch.nn.functional as F
from cirloss import *

def train(train_loader, model, optimizer, device, epoch=0):
    model.train()
    train_losses = []
    criterion = CircleLoss(m=0, gamma=1)
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device, dtype=torch.float)
        labels = labels.long().to(device)
        optimizer.zero_grad()
        outputs = model(data)
        inp_sp, inp_sn = convert_label_to_similarity(outputs[1], labels)
        #0.4 128 0.25 256 0.1 256
        loss = criterion(inp_sp, inp_sn)
        #loss = F.cross_entropy(outputs[0], labels)

        loss.backward()
        train_losses.append(loss.item())

        optimizer.step()
        if i % 1000 == 0:
            print('Train Epoch :{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1, i * len(data), len(train_loader.dataset), 100. * i / len(train_loader), loss.item()))
    return torch.tensor(train_losses).mean()
