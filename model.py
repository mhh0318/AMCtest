# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/7/1 22:11
@author: merci
"""
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1):  # 3 and 5
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
        self.clf = nn.Sequential(
            nn.Linear(512, 128),
            nn.SELU(True),
            nn.AlphaDropout(0.5),
            nn.Linear(128, 128),
            nn.SELU(True),
            nn.AlphaDropout(0.5),
            nn.Linear(128, 11)
        )
        '''
        self.clf = nn.Linear(32,12)

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
        x = self.mp(self.layer11(x))  #128*32

        x = x.view(x.size(0), -1)

        code = x

        x = self.clf(x)

        return x, nn.functional.normalize(code)
