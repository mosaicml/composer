# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The CIFAR ResNet torch module.

See the :doc:`Model Card </model_cards/resnet>` for more details.
"""

# Code below adapted from https://github.com/facebookresearch/open_lth
# and https://github.com/pytorch/vision

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from composer.models import Initializer

__all__ = ['ResNetCIFAR', 'ResNet9']


class ResNetCIFAR(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample: bool = False):
            super(ResNetCIFAR.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)
            self.relu = nn.ReLU(inplace=True)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out),
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x: torch.Tensor):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu(out)

    def __init__(self, plan: List[Tuple[int, int]], initializers: List[Initializer], outputs: int = 10):
        super(ResNetCIFAR, self).__init__()
        outputs = outputs or 10

        self.num_classes = outputs

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)
        self.relu = nn.ReLU(inplace=True)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNetCIFAR.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        for initializer in initializers:
            initializer = Initializer(initializer)
            self.apply(initializer.get_initializer())

    def forward(self, x: torch.Tensor):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def is_valid_model_name(model_name: str):
        valid_model_names = [f'resnet_{layers}' for layers in (20, 56)]
        return (model_name in valid_model_names)

    @staticmethod
    def get_model_from_name(model_name: str, initializers: List[Initializer], outputs: int = 10):
        """The naming scheme for a ResNet is ``'resnet_D[_W]'``.

        D is the model depth (e.g. ``'resnet_56'``)
        """

        if not ResNetCIFAR.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        depth = int(model_name.split('_')[-1])  # for resnet56, depth 56, width 16
        if len(model_name.split('_')) == 2:
            width = 16
        else:
            width = int(model_name.split('_')[3])

        if (depth - 2) % 3 != 0:
            raise ValueError('Invalid ResNetCIFAR depth: {}'.format(depth))
        num_blocks = (depth - 2) // 6

        model_arch = {
            56: [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)],
            20: [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)],
        }

        return ResNetCIFAR(model_arch[depth], initializers, outputs)


# adapted from https://raw.githubusercontent.com/matthias-wright/cifar10-resnet/master/model.py
# under the MIT license
class ResNet9(nn.Module):
    """A 9-layer residual network, excluding BatchNorms and activation functions.

    Based on the myrtle.ai `blog`_ and Deep Residual Learning for Image Recognition (`He et al, 2015`_).

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.

    .. _blog: https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
    .. _He et al, 2015: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(inplanes=128, planes=128, stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(inplanes=256, planes=256, stride=1),
        )

        self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.body(x)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
