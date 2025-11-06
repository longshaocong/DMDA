import torch 
import torch.nn as nn


class feat_classifier(nn.Module):
    def __init__(self, classes, feat_dim) -> None:
        super(feat_classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, classes)

    def forward(self, x):
        return self.fc(x)

class expert_classifier(nn.Module):
    def __init__(self, classes, feat_dim, mid=512):
        super(expert_classifier, self).__init__()
        self.cp = nn.Sequential(
            nn.Linear(feat_dim, mid, bias=False), 
            nn.BatchNorm1d(mid), 
            nn.ReLU(inplace=True), 
            nn.Linear(mid, mid // 4, bias=False), 
            nn.BatchNorm1d(mid // 4), 
            nn.ReLU(inplace=True), 
            nn.Linear(mid // 4, classes, bias=False)
        )

    def forward(self, x):
        return self.cp(x)


class semantic_embedding(nn.Module):
    def __init__(self, classes, feat_dim) -> None:
        super(semantic_embedding, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(classes, feat_dim)
        )

    def forward(self, x):
        return self.layer(x)