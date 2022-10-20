'''
script to predict the label
'''

from turtle import forward
import torch 
import torch.nn as nn


class feat_classifier(nn.Module):
    def __init__(self, classes, feat_dim) -> None:
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, classes)

    def forward(self, x):
        return self.fc(x)

class class_embedding(nn.Module):
    def __init__(self, classes, feat_dim) -> None:
        super(class_embedding, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(classes, feat_dim)
        )

    def forward(self, x):
        return self.layer(x)

class projector(nn.Module):
    def __init__(self, prev_dim, feat_dim):
        super(projector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False), 
            nn.BatchNorm1d(prev_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(prev_dim, prev_dim, bias=False), 
            nn.BatchNorm1d(prev_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(prev_dim, feat_dim, bias=False), # simsiam中直接使用的resent的fc, 其resnet定义中zero_init_residual=True
            nn.BatchNorm1d(feat_dim, affine=False)
        )

    def forward(self, x):
        return self.fc(x)


class predictor(nn.Module):
    def __init__(self, feat_dim, pred_dim):
        super(predictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, pred_dim, bias=False), 
            nn.BatchNorm1d(pred_dim), 
            nn.ReLU(inplace=True), 
            nn.Linear(pred_dim, feat_dim)
        )

    def forward(self, x):
        return self.fc(x)
