# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch
from torch import nn as nn


def round_channels(
    channels: float,
    width_multiplier: float,
    divisor: int = 8,
    min_value: Optional[int] = None,
) -> int:
    """Round number of channels after scaling with width multiplier.

    This function ensures that channel integers halfway in-between divisors is rounded up.

    Args:
        channels (float): Number to round.
        width_multiplier (float): Amount to scale `channels`.
        divisor (int): Number to make the output divisible by.
        min_value (int, optional): Minimum value the output can be. If not specified, defaults
            to the ``divisor``.
    """
    if not width_multiplier:
        return int(channels)
    channels *= width_multiplier

    min_value = min_value or divisor
    new_channels = max(min_value, int(channels + divisor / 2) // divisor * divisor)
    if new_channels < 0.9 * channels:  # increase channels if rounding decreases by >10%
        new_channels += divisor
    return new_channels


def calculate_same_padding(kernel_size, dilation, stride):
    """Calculates the amount of padding to use to get the "SAME" functionality in Tensorflow."""
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def drop_connect(inputs: torch.Tensor, drop_connect_rate: float, training: bool):
    """Randomly mask a set of samples. Provides similar regularization as stochastic depth.

    Args:
        input (torch.Tensor): Input tensor to mask.
        drop_connect_rate (float): Probability of droppping each sample.
        training (bool): Whether or not the model is training
    """
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    rand_tensor = keep_prob + torch.rand(
        [inputs.size()[0], 1, 1, 1],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    rand_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * rand_tensor
    return output


class SqueezeExcite(nn.Module):
    """Squeeze Excite Layer.

    Args:
        in_channels (int): Number of channels in the input tensor.
        latent_channels (int): Number of hidden channels.
        act_layer (torch.nn.Module): Activation layer to use in block.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(latent_channels, in_channels, kernel_size=1, bias=True)
        self.gate_fn = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        out = self.global_avg_pool(x)
        out = self.conv_reduce(out)
        out = self.act1(out)
        out = self.conv_expand(out)
        out = x * self.gate_fn(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution layer.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        se_ratio (float): How much to scale `in_channels` for the hidden layer
            dimensionality of the squeeze-excite module.
        drop_connect_rate (float): Probability of dropping a sample before the
            identity connection, provides regularization similar to stochastic
            depth.
        act_layer (torch.nn.Module): Activation layer to use in block.
        norm_kwargs (dict): Normalization layer's keyword arguments.
        norm_layer (torch.nn.Module): Normalization layer to use in block.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 se_ratio: float,
                 drop_connect_rate: float,
                 act_layer: Callable[..., nn.Module],
                 norm_kwargs: dict,
                 norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.has_residual = (in_channels == out_channels and stride == 1)
        self.has_se = se_ratio is not None and se_ratio > 0.0

        padding = calculate_same_padding(kernel_size, dilation=1, stride=stride)
        self.conv_depthwise = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels,
                                        groups=in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=False)
        self.bn1 = norm_layer(in_channels, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        if self.has_se:
            latent_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcite(in_channels, latent_channels, act_layer)

        self.conv_pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = norm_layer(out_channels, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

    def forward(self, input: torch.Tensor):
        residual = input

        out = self.conv_depthwise(input)
        out = self.bn1(out)
        out = self.act1(out)

        if self.has_se:
            out = self.se(out)

        out = self.conv_pointwise(out)
        out = self.bn2(out)
        out = self.act2(out)

        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                out = drop_connect(out, self.drop_connect_rate, self.training)
            out += residual
        return out


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    This block is implemented as as defined in
    `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`_ (Sandler et al, 2018).

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        expand_ratio (int): How much to expand the input channels for the
            depthwise convolution.
        se_ratio (float): How much to scale `in_channels` for the hidden layer
            dimensionality of the squeeze-excite module.
        drop_connect_rate (float): Probability of dropping a sample before the
            identity connection, provides regularization similar to stochastic
            depth.
        act_layer (torch.nn.Module): Activation layer to use in block.
        norm_kwargs (dict): Normalization layer's keyword arguments.
        norm_layer (torch.nn.Module): Normalization layer to use in block.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 expand_ratio: int,
                 se_ratio: float,
                 drop_connect_rate: float,
                 act_layer: Callable[..., nn.Module],
                 norm_kwargs: dict,
                 norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.has_residual = (in_channels == out_channels and stride == 1)
        self.has_se = se_ratio is not None and se_ratio > 0.0

        mid_channels = round_channels(in_channels, expand_ratio)

        # Point-wise convolution expansion
        self.conv1x1_expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(mid_channels, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise Convolution
        padding = calculate_same_padding(kernel_size, dilation=1, stride=stride)
        self.conv_depthwise = nn.Conv2d(in_channels=mid_channels,
                                        out_channels=mid_channels,
                                        groups=mid_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        bias=False)
        self.bn2 = norm_layer(mid_channels, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze and Excitation layer, if specified
        if self.has_se:
            latent_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcite(mid_channels, latent_channels, act_layer)

        # Point-wise convolution contraction
        self.conv1x1_contract = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels, **norm_kwargs)

    def forward(self, input: torch.Tensor):
        residual = input

        out = self.conv1x1_expand(input)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv_depthwise(out)
        out = self.bn2(out)
        out = self.act2(out)

        if self.has_se:
            out = self.se(out)

        out = self.conv1x1_contract(out)
        out = self.bn3(out)

        if self.has_residual:
            if self.drop_connect_rate:
                out = drop_connect(out, self.drop_connect_rate, self.training)
            out += residual
        return out
