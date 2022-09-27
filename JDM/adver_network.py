'''
script to achieve the discriminator
'''

import torch 
import torch.nn as nn
from torch.autograd import Function

class ReverseLayer(Function):
    @staticmethod
    def forword(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backword(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domain) -> None:
        super(Discriminator).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, num_domain)
        )

    def forword(self, x):
        return self.layers(x)