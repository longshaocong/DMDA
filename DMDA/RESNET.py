'''
the script to implement the bottle network, e.g., resnet
'''

import torch 
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torchvision.models

res_dict = {'resnet18': models.resnet18, 'resnet50': models.resnet50}

class RES(nn.Module):
    
    def __init__(self, res_name):
        super(RES, self).__init__()
        model = res_dict[res_name](pretrained=True)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.in_features = model.fc.in_features
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        map = self.layer4(x)
        x = self.avgpool(map)
        x = x.view(x.size(0), -1)
        return x, map

def get_fea(args):
    net = RES(args.net)
    return net