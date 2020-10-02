#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/1 22:27
@author: merci
"""
import torch
import torch.nn.functional as F

def test(test_loader, model, device):
    model.eval()
    test_loss = 0
    correct = 0
    predictions = []
    probability = []
    for data, labels in test_loader:
        data = data.to(device, dtype=torch.float)
        labels = labels.long().to(device)

        output = model(data)
        probability.extend(output[0].cpu().detach().numpy())
        prob = F.softmax(output[0], dim=1)
        # test_loss += F.cross_entropy(output[0],labels).item()
        pred = torch.max(output[0], 1)[1]
        correct += pred.eq(labels.data.view_as(pred)).sum()
        predictions.extend(pred.cpu().numpy())
    acc = 100. * correct / len(test_loader.dataset)
    # test_loss /= len(test_loader.dataset)
    print('\nTest set:  Accuracy: {}/{} ({:.2f}%)\t'.format(
         correct, len(test_loader.dataset), acc))
    return predictions, probability, acc