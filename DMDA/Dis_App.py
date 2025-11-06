'''
script to achieve the distribution approximator
'''

import torch 
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Any, Tuple
import numpy as np

class Distribution(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_domain) -> None:
        super(Distribution, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, num_domain)
        )

    def forward(self, x):
        return self.layers(x)


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
