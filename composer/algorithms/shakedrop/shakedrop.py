# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.autograd import Function


class ShakeDropFunction(Function):
    '''See :class:`~composer.algorithms.shakedrop.shakedrop:ShakeDrop`.'''

    @staticmethod
    def forward(ctx,
                x: Tensor,
                training: bool = True,
                drop_prob: float = 0.5,
                alpha_min: float = -1,
                alpha_max: float = 1) -> Tensor:
        if not training:
            return (1 - drop_prob) * x

        gate = torch.empty(1, device=x.device).bernoulli_(1 - drop_prob)
        ctx.save_for_backward(gate)
        if gate:
            return x

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        alpha = torch.empty(shape, device=x.device).uniform_(alpha_min, alpha_max)
        return alpha * x

    @staticmethod
    def backward(ctx, dy) -> Tuple[Tensor, None, None, None, None]:
        gate = ctx.saved_tensors[0]
        if gate:
            return dy, None, None, None, None

        shape = (dy.shape[0],) + (1,) * (dy.ndim - 1)
        beta = torch.rand(shape, device=dy.device)
        return beta * dy, None, None, None, None


class ShakeDrop(nn.Module):
    '''ShakeDrop <https://arxiv.org/abs/1802.02375> is a regularization method that stochastically
    scales the gradients.

    Args:
        drop_prob (float): Drop probability
        alpha_min (float): Minimum of alpha range
        alpha_max (float): Maximum of alpha range
    '''

    def __init__(self, drop_prob: float = 0.5, alpha_min: float = -1, alpha_max: float = 1):
        super().__init__()
        self.drop_prob = drop_prob
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def forward(self, x: Tensor) -> Tensor:
        return ShakeDropFunction.apply(x, self.training, self.drop_prob, self.alpha_min, self.alpha_max)
