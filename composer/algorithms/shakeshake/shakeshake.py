import torch
from torch.autograd import Function
from torch import nn
from torch import Tensor
from typing import Tuple


class ShakeShakeFunction(Function):
    '''See `ShakeShake`.'''

    @staticmethod
    def forward(ctx, x1: Tensor, x2: Tensor, training: bool = True) -> Tensor:
        if training:
            shape = (x1.shape[0],) + (1,) * x1.shape[1:]
            alpha = torch.rand(shape, device=x.device)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[Tensor, Tensor, None]:
        shape = (x1.shape[0],) + (1,) * x1.shape[1:]
        beta = torch.rand(shape, device=dy.device)
        return beta * dy, (1 - beta) * dy, None


class ShakeShake(nn.Module):
    '''`ShakeShake <https://arxiv.org/abs/1705.07485>`_ is a regularization method that replaces
    summation of parallel branches with stochastic affine combination.
    '''

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return ShakeShakeFunction.apply(x1, x2, self.training)
