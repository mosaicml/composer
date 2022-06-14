# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet model.

Adapted from `(Generic) EfficientNets for PyTorch. <https://github.com/rwightman/gen-efficientnet-pytorch>`_.
"""

import math
import re
from typing import Callable, Optional

import torch
import torch.nn as nn

from composer.models.efficientnetb0._layers import (DepthwiseSeparableConv, MBConvBlock, calculate_same_padding,
                                                    round_channels)

__all__ = ['EfficientNet']


class EfficientNet(nn.Module):
    """EfficientNet model based on (`Tan et al, 2019 <https://arxiv.org/abs/1905.11946>`_).

    Args:
        num_classes (int): Size of the EfficientNet output, typically viewed
             as the number of classes in a classification task.
        width_multiplier (float, optional): How much to scale the EfficientNet-B0 channel
             dimension throughout the model. Default: ``1.0``.
        depth_multiplier (float, optional): How much to scale the EFficientNet-B0 depth. Default: ``1.0``.
        drop_rate (float, optional): Dropout probability for the penultimate activations. Default: ``0.2``.
        drop_connect_rate (float, optional): Probability of dropping a sample before the
             identity connection, provides regularization similar to stochastic
             depth. Default: ``0.2``.
        act_layer (torch.nn.Module, optional): Activation layer to use in the model. Default: ``nn.SiLU``.
        norm_kwargs (dict, optional): Normalization layer's keyword arguments. Default: ``{"momentum": 0.1, "eps": 1e-5}``.
        norm_layer (torch.nn.Module, optional): Normalization layer to use in the model. Default: ``nn.BatchNorm2d``.
    """

    # EfficientNet-B0 architecture specification.
    # block_strings are decoded into block level hyperparameters.
    # r=repeat, k=kernel_size, s=stride, e=expand_ratio, i=in_channels, o=out_channels, se=se_ratio.
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
                 norm_kwargs: Optional[dict] = None,
                 norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d):
        super(EfficientNet, self).__init__()
        self.num_classes = num_classes

        if norm_kwargs is None:
            norm_kwargs = {'momentum': 0.1, 'eps': 1e-5}

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
            block_args['in_channels'] = round_channels(
                block_args['in_channels'],
                width_multiplier,
            )
            block_args['out_channels'] = round_channels(
                block_args['out_channels'],
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
        """Instantiate an EfficientNet model family member based on the model_name string.

        Args:
            model_name: (str): One of ``'efficientnet-b0'`` through ``'efficientnet-b7'``.
            num_classes (int): Size of the EfficientNet output, typically viewed as the number of classes in a classification task.
            drop_connect_rate (float): Probability of dropping a sample before the identity connection,
                provides regularization similar to stochastic depth.
        """

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
        block_args = {
            'kernel_size': int(args['k']),
            'stride': int(args['s']),
            'expand_ratio': int(args['e']),
            'in_channels': int(args['i']),
            'out_channels': int(args['o']),
            'se_ratio': float(args['se']) if 'se' in args else None,
        }
        return block_args, num_repeat
