# Copyright 2021 MosaicML. All Rights Reserved.

# Code adapted from (Generic) EfficientNets for PyTorch repo:
# https://github.com/rwightman/gen-efficientnet-pytorch
import math
import re
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


def round_channels(
    channels: float,
    width_multiplier: float,
    divisor: int = 8,
    min_value: Optional[int] = None,
) -> int:
    """Round number of channels after scaling with width multiplier. This function ensures that channel integers halfway
    inbetween divisors is rounded up.

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
    """Mobile Inverted Residual Bottleneck Block as defined in https://arxiv.org/abs/1801.04381.

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


class EfficientNet(nn.Module):
    """EfficientNet architecture designed for ImageNet in https://arxiv.org/abs/1905.11946.

    Args:
        num_classes (int): Size of the EfficientNet output, typically viewed
             as the number of classes in a classification task.
        width_multiplier (float): How much to scale the EfficientNet-B0 channel
             dimension throughout the model.
        depth_multiplier (float): How much to scale the EFficientNet-B0 depth.
        drop_rate (float): Dropout probability for the penultimate activations.
        drop_connect_rate (float): Probability of dropping a sample before the
             identity connection, provides regularization similar to stochastic
             depth.
        act_layer (torch.nn.Module): Activation layer to use in the model.
        norm_kwargs (dict): Normalization layer's keyword arguments.
        norm_layer (torch.nn.Module): Normalization layer to use in the model.
    """

    # EfficientNet-B0 architecture specification
    _blocks_strings = [
        'r1_k3_s1_e1_i32_o16_se0.25',
        'r2_k3_s2_e6_i16_o24_se0.25',
        'r2_k5_s2_e6_i24_o40_se0.25',
        'r3_k3_s2_e6_i40_o80_se0.25',
        'r3_k5_s1_e6_i80_o112_se0.25',
        'r4_k5_s2_e6_i112_o192_se0.25',
        'r1_k3_s1_e6_i192_o320_se0.25',
    ]

    def __init__(self,
                 num_classes: int,
                 width_multiplier: float = 1.0,
                 depth_multiplier: float = 1.0,
                 drop_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 act_layer: Callable[..., nn.Module] = nn.SiLU,
                 norm_kwargs: dict = {
                     "momentum": 0.1,
                     "eps": 1e-5
                 },
                 norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes

        in_channels = 3
        out_channels = round_channels(32, width_multiplier)
        padding = calculate_same_padding(kernel_size=3, dilation=1, stride=2)
        self.conv_stem = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=padding,
            bias=False,
        )
        self.bn1 = norm_layer(num_features=out_channels, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Count the number of blocks in the model
        block_count = 0.
        for block_string in self._blocks_strings:
            _, num_repeat = self._decode_block_string(block_string)
            block_count += num_repeat

        # Decode block strings and add blocks
        block_idx = 0.
        blocks = []
        block_args = {}
        for block_string in self._blocks_strings:
            block_args, num_repeat = self._decode_block_string(block_string)
            # Scale channels and number of repeated blocks based on multipliers
            block_args["in_channels"] = round_channels(
                block_args["in_channels"],
                width_multiplier,
            )
            block_args["out_channels"] = round_channels(
                block_args["out_channels"],
                width_multiplier,
            )
            num_repeat = int(math.ceil(depth_multiplier * num_repeat))

            # Add activation, normalization layers, and drop connect
            block_args['act_layer'] = act_layer
            block_args['norm_kwargs'] = norm_kwargs
            block_args['norm_layer'] = norm_layer

            # Delete expand_ratio when set to 1 to use depthwise separable convolution layer
            if block_args['expand_ratio'] == 1:
                del block_args['expand_ratio']

            for i in range(num_repeat):
                # Linearly decay drop_connect_rate across model depth
                block_args['drop_connect_rate'] = drop_connect_rate * block_idx / block_count

                if 'expand_ratio' not in block_args:
                    blocks.append(DepthwiseSeparableConv(**block_args))
                else:
                    blocks.append(MBConvBlock(**block_args))
                block_idx += 1

                # Only the first block in a stage can have stride != 1
                if i == 0:
                    block_args['stride'] = 1
                    block_args['in_channels'] = block_args['out_channels']

        self.blocks = nn.Sequential(*blocks)

        in_channels = block_args['out_channels']
        out_channels = round_channels(1280, width_multiplier)
        self.conv_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = norm_layer(out_channels, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(out_channels, num_classes)

        # Initialization from gen-efficientnet-pytorch repo
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan_out = (m.kernel_size[0] * m.kernel_size[1] * m.out_channels) // m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

    def extract_features(self, input: torch.Tensor):
        out = self.conv_stem(input)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.blocks(out)
        out = self.conv_head(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.global_avg_pool(out)
        return out.flatten(1)

    def forward(self, input: torch.Tensor):
        out = self.extract_features(input)
        out = self.dropout(out)
        return self.classifier(out)

    @staticmethod
    def get_model_from_name(model_name: str, num_classes, drop_connect_rate: float):
        """Instantiate an EfficientNet model family member based on the model_name string."""

        # Coefficients: width, depth, res, dropout
        model_arch = {
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        }

        model_params = model_arch[model_name]
        width_multiplier = model_params[0]
        depth_multiplier = model_params[1]
        drop_rate = model_params[3]
        return EfficientNet(num_classes=num_classes,
                            width_multiplier=width_multiplier,
                            depth_multiplier=depth_multiplier,
                            drop_rate=drop_rate,
                            drop_connect_rate=drop_connect_rate)

    def _decode_block_string(self, block_string: str):
        """Decodes an EfficientNet block specification string into a dictionary of keyword arguments for a block in the
        architecture."""

        arg_strings = block_string.split('_')
        args = {}
        for arg_string in arg_strings:
            splits = re.split(r'(\d.*)', arg_string)
            if len(splits) >= 2:
                key, value = splits[:2]
                args[key] = value
        num_repeat = int(args['r'])
        block_args = dict(kernel_size=int(args['k']),
                          stride=int(args['s']),
                          expand_ratio=int(args['e']),
                          in_channels=int(args['i']),
                          out_channels=int(args['o']),
                          se_ratio=float(args['se']) if 'se' in args else None)  # type: Dict[str, Any]
        return block_args, num_repeat
