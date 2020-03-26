#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/1/8 14:53
@author: merci
"""
import torch
from torch.utils import data
from AMC_Features import *
#from ResAMC import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE


def plot_cm(cm, title='Confusion matrix', cmap=plt.cm.Reds,  labels=[]):
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.axis('tight')
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, fontsize=20)
    plt.yticks(tick_marks, labels,fontsize=20)
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
            plt.text(j - 0.3, i, textstr[i, j]+'%', size='xx-large' ,color=ijcolor)



def visualize(new_index, features, labels, figname):
    t_sne = TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = t_sne.fit_transform(features[new_index])
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(25, 15))
    for i in range(X_norm.shape[0]):
        #plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[new_index][i]), color=plt.cm.tab20b(labels[new_index][i]),
                 #fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(X_norm[i, 0], X_norm[i, 1], alpha=0.7, color=plt.cm.tab20b(labels[new_index][i]))
    plt.xticks([])
    plt.yticks([])
    plt.savefig('/home/hu/{}.png'.format(figname))
    plt.figure(figsize=(25, 15))
    for i in np.unique(labels[new_index]):
        for k in X_norm[np.where(labels[new_index]==i)]:
            plt.subplot(3, 5, i + 1)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.scatter(k[0],k[1],alpha=0.7,color=plt.cm.tab20b(i))
    plt.show()


def cluster_train(model,dataloader):
    model = model
    model.eval()
    l_code = []
    label = dataloader.dataset.label.numpy()
    for signal, labels in dataloader:
        signal = signal.to(device, dtype=torch.float)
        output = model(signal)
        l_code.append(output[0].detach().cpu().numpy()) ############
    l = np.concatenate(l_code)
    num_ = np.unique(label)
    # node = np.zeros([len(num_), 4096])#############
    node = np.zeros([len(num_), 11])
    threshold = np.zeros(11)
    cosine = np.zeros(11)
    cm = np.zeros((11,11,11))
    for k in num_:
        label[label == k] = np.squeeze(np.argwhere(num_ == k))
    for i in range(11):
        mask = np.where(label == i)
        l_tmp = l[mask].reshape(l[mask].shape[0], -1)
        node_ = np.mean(l_tmp, axis=0)
        node[i] = node_
        # l2 distance
        distance = np.sum(np.square(l_tmp - node_), axis=1)
        #distance = np.linalg.norm((l_tmp - node_), ord=2, axis=1)
        # l1 distance
        #distance = np.linalg.norm((l_tmp - node_), ord=1, axis=1)
        # consine similarity
        angle = np.dot(l_tmp, node_) /(np.linalg.norm(l_tmp,axis=1)*np.linalg.norm(node_))
        threshold_ = np.percentile(distance, 95)
        angle_ = np.percentile(angle, 95)
        threshold[i] = threshold_
        cosine[i] = angle_
        cm[i] = np.cov(l_tmp.T)
    print('TrainFinish')
    return node, threshold, cosine, cm

def cluster_test(model,dataloader,node,threshold,cosine):
    model = model
    model.eval()
    l_code = []
    label = dataloader.dataset.label.numpy()
    for signal, labels in dataloader:
        signal = signal.to(device, dtype=torch.float)
        output = model(signal)
        l_code.append(output[0].detach().cpu().numpy()) ####################
    l = np.concatenate(l_code)
    l = l.reshape(l.shape[0], -1)
    num_ = np.unique(label)
    result_dist = np.full((11,l.shape[0]),np.inf)
    result_angle = np.full((11,l.shape[0]),np.inf)

    for k in num_:
        label[label == k] = np.squeeze(np.argwhere(num_ == k))
    for i in range(11):
        distance_ = np.sum(np.square(l - node[i]), axis=1)
        #distance_ = np.linalg.norm((l - node[i]), ord=1, axis=1)
        angle_ = np.dot(l, node[i]) /(np.linalg.norm(l, axis=1)*np.linalg.norm(node[i]))
        #mask0 = np.where(distance_<=threshold[i])
        #mask1 = np.where(angle_ >= cosine[i])

        result_dist[i] = distance_
        result_angle[i] = angle_
    '''
    new_signal_dist = []
    for i,j in enumerate(result_dist.T):
        if np.isinf(j).all():
            new_signal_dist.append(i)
    exist_signal_dist = np.setdiff1d(np.arange(l.shape[0]), np.array(new_signal_dist))
    '''
    predict_dict = np.argmin(result_dist, axis=0)
    predict_angle = np.argmax(result_angle, axis=0)
    # predict = np.where(predict_dict == predict_angle)

    a = result_dist[predict_dict, range(372103)]
    b = result_angle[predict_angle, range(372103)]
    ta = threshold[predict_dict]
    tb = cosine[predict_angle]
    predict_index = np.intersect1d(np.where(b >= tb-0.05), np.where(a < ta))
    '''
    angle_select=np.where(result_angle[predict,exist_signal_dist]>=(cosine[predict]-0.03))
    new_signal_dist = np.setdiff1d(np.arange(l.shape[0]),angle_select)
    '''
    asnew = np.setdiff1d(np.arange(l.shape[0]), predict_index)
    exist0 = np.intersect1d(np.where((label!=6)&(label!=12)&(label!=13)),predict_index)

    y = label[exist0]
    lable_list = np.unique(y)
    for i in lable_list:
        y[y == i] = np.squeeze(np.argwhere(lable_list == i))
    accuracy = accuracy_score(y, predict_angle[exist0])
    print(accuracy_score(y, predict_angle[exist0]))

    print('\nxxxxxxxxxxxxx')
    for i in range(13):
        print(np.intersect1d(np.where((label == i)), asnew).shape)
    return accuracy

if __name__ == '__main__':
    start = time.time()
    test_loader = data.DataLoader(dataset=LongerSequence(train=True,unknown=2), batch_size=128, shuffle=False)
    new_loader = data.DataLoader(dataset=LongerSequence(train=False,unknown=0), batch_size=128, shuffle=False)
    #train_loader = data.DataLoader(dataset=LongerSequence(train=True), batch_size=128, shuffle=True)
    #test_loader = data.DataLoader(dataset=HDF5Data(train=False), batch_size=128, shuffle=False)
    #train_loader = data.DataLoader(dataset=HDF5Data(train=True), batch_size=128, shuffle=True)
    model_l = torch.load('/home/hu/AMCD3/Linear/2048-new11.ckpt').to(device)
    #model_l = torch.load('/home/hu/AMCpri/Aug/14.ckpt').to(device)
    node,threshold,cosine = cluster_train(model_l, test_loader)
    result = cluster_test(model_l, new_loader, node, threshold,cosine)
    total = time.time()-start
    print('Time: {}m {}s'.format(total//60,total%60))

