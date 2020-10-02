#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/1 22:29
@author: merci
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_cm(cm, title='Confusion matrix', cmap=plt.cm.Reds,  labels=[]):
    plt.figure(figsize=(15,15))
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
            plt.text(j - 0.2, i, textstr[i, j] + '%', size='xx-large' ,color=ijcolor)


def adjust_learning_rate(optimizer, epoch):
    if epoch < 150:
        lr = 0.001
    elif epoch < 225:
        lr = 0.001 * 0.1
    else:
        lr = 0.001 * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
