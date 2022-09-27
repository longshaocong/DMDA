'''
script to predict the label
'''

from turtle import forward
import torch 
import torch.nn as nn


class feat_classifier(nn.Module):
    def __init__(self, classes, feat_dim) -> None:
        super(feat_classifier).__init__()
        self.fc = nn.Linear(feat_dim, classes)

    def forward(self, x):
        return self.fc(x)

class class_embedding(nn.Module):
    def __init__(self, classes, feat_dim) -> None:
        super(class_embedding).__init__()
        self.layer = nn.Sequential(
            nn.Linear(classes, feat_dim //4), 
            nn.BatchNorm1d(feat_dim //4), 
            nn.ReLU(), 
            nn.Linear(feat_dim //4, feat_dim)
        )

    def forward(self, x):
        return self.layer(x)