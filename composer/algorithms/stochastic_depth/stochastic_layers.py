# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Stochastic forward functions for ResNet Bottleneck modules."""

from typing import Optional

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torchvision.models.resnet import Bottleneck

__all__ = ['make_resnet_bottleneck_stochastic', 'BlockStochasticModule']


def block_stochastic_forward(self, x):
    """ResNet Bottleneck forward function where the layers are randomly
        skipped with probability ``drop_rate`` during training.
    """

    identity = x

    sample = (not self.training) or bool(torch.bernoulli(1 - self.drop_rate))

    if sample:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if not self.training:
            out = out * (1 - self.drop_rate)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
    else:
        if self.downsample is not None:
            out = self.relu(self.downsample(identity))
        else:
            out = identity
    return out


def _sample_drop(x: torch.Tensor, sample_drop_rate: float, is_training: bool):
    """Randomly drops samples from the input batch according to the `sample_drop_rate`.

    This is implemented by setting the samples to be dropped to zeros.
    """

    keep_probability = (1 - sample_drop_rate)
    if not is_training:
        return x * keep_probability
    rand_dim = [x.shape[0]] + [1] * len(x.shape[1:])
    sample_mask = keep_probability + torch.rand(rand_dim, dtype=x.dtype, device=x.device)
    sample_mask.floor_()  # binarize
    x *= sample_mask
    return x


def sample_stochastic_forward(self, x):
    """ResNet Bottleneck forward function where samples are randomly
        dropped with probability ``drop_rate`` during training.
    """

    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    if self.drop_rate:
        out = _sample_drop(out, self.drop_rate, self.training)
    out += identity

    return self.relu(out)


def make_resnet_bottleneck_stochastic(
    module: Bottleneck,
    module_index: int,
    module_count: int,
    drop_rate: float,
    drop_distribution: str,
    stochastic_method: str,
):
    """Model surgery policy that dictates how to convert a ResNet Bottleneck layer into a stochastic version.
    """

    if drop_distribution == 'linear':
        drop_rate = ((module_index + 1) / module_count) * drop_rate
    module.drop_rate = torch.tensor(drop_rate)

    stochastic_func = block_stochastic_forward if stochastic_method == 'block' else sample_stochastic_forward
    module.forward = stochastic_func.__get__(module)  # Bind new forward function to ResNet Bottleneck Module

    return module


class BlockStochasticModule(nn.Module):
    """A convenience class that stochastically executes the provided main path of a residual block.

    Args:
        main (GraphModule): Operators in the main (non-residual) path of a residual block.
        residual (GraphModule | None): Operators, if any, in the residual path of a residual block.
        drop_rate: The base probability of dropping this layer. Must be between 0.0 (inclusive) and 1.0 (inclusive).

    Returns:
        BlockStochasticModule: An instance of :class:`.BlockStochasticModule`.
    """

    def __init__(self, main: GraphModule, residual: Optional[GraphModule] = None, drop_rate: float = 0.2):
        super().__init__()
        self.drop_rate = torch.tensor(drop_rate)
        self.main = main
        self.residual = residual

    def forward(self, x):
        sample = (not self.training) or bool(torch.bernoulli(1 - self.drop_rate))
        # main side is the non-residual connection
        residual_result = x
        # residual side may or may not have any operations
        if self.residual:
            residual_result = self.residual(x)

        if sample:
            main_result = self.main(x)
            if not self.training:
                main_result = main_result * (1 - self.drop_rate)
            residual_result = torch.add(main_result, residual_result)
        return residual_result
