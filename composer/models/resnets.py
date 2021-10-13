# Copyright 2021 MosaicML. All Rights Reserved.

# Code below adapted from https://github.com/facebookresearch/open_lth
# and https://github.com/pytorch/vision

# type: ignore
# yapf: disable
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet
from torchvision.models.resnet import conv1x1, conv3x3

from composer.models import Initializer


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        out += identity
        out = self.relu(out)

        return out


class Torchvision_ResNet(ResNet):

    def __init__(self, block, layers, num_classes=1000, width=64):
        """To make it possible to vary the width, we need to
        override the constructor of the torchvision resnet."""

        nn.Module.__init__(self)  # Skip the parent constructor. This replaces it.
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The subsequent blocks.
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width * 2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, width * 4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, width * 8, layers[3], stride=2, dilate=False)

        # The last layers.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8 * block.expansion, num_classes)

        # Default init.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ImageNet_ResNet(nn.Module):
    """A residual neural network as originally designed for ImageNet."""

    def __init__(self, model_fn, initializers: List[Union[str, Initializer]], outputs=None):
        super(ImageNet_ResNet, self).__init__()

        self.num_classes = outputs
        self.model = model_fn(num_classes=outputs or 1000)
        self.criterion = nn.CrossEntropyLoss()
        for initializer in initializers:
            initializer = Initializer(initializer)
            self.apply(initializer.get_initializer())

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('imagenet_resnet_') and 4 >= len(model_name.split('_')) >= 3 and
                model_name.split('_')[2].isdigit() and int(model_name.split('_')[2]) in [18, 34, 50, 101, 152, 200])

    @staticmethod
    def get_model_from_name(model_name, initializers, outputs=1000):
        """Name: imagenet_resnet_D[_W].
        D is the model depth (e.g., 50 for ResNet-50).
        W is the model width - the number of filters in the first
        residual layers. By default, this number is 64."""

        model_arch = {
            18: {
                'block': BasicBlock,
                'layers': [2, 2, 2, 2]
            },
            34: {
                'block': BasicBlock,
                'layers': [3, 4, 6, 3]
            },
            50: {
                'block': Bottleneck,
                'layers': [3, 4, 6, 3]
            },
            101: {
                'block': Bottleneck,
                'layers': [3, 4, 23, 3]
            },
            152: {
                'block': Bottleneck,
                'layers': [3, 8, 36, 3]
            },
            200: {
                'block': Bottleneck,
                'layers': [3, 24, 36, 3]
            },
            269: {
                'block': Bottleneck,
                'layers': [3, 30, 48, 8]
            },
        }

        if not ImageNet_ResNet.is_valid_model_name(model_name):
            raise ValueError(f'Invalid model name: {model_name}')

        num = int(model_name.split('_')[2])
        block = model_arch[num]['block']
        layers = model_arch[num]['layers']
        model_fn = partial(Torchvision_ResNet, block, layers)
        if len(model_name.split('_')) == 4:
            width = int(model_name.split('_')[3])
            model_fn = partial(model_fn, width=width)

        return ImageNet_ResNet(model_fn, initializers, outputs)


class CIFAR_ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(CIFAR_ResNet.Block, self).__init__()

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

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return self.relu(out)

    def __init__(self, plan, initializers: List[Union[str, Initializer]], outputs=None):
        super(CIFAR_ResNet, self).__init__()
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
                blocks.append(CIFAR_ResNet.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        for initializer in initializers:
            initializer = Initializer(initializer)
            self.apply(initializer.get_initializer())

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_resnet_') and 4 >= len(model_name.split('_')) >= 3 and
                model_name.split('_')[2].isdigit() and int(model_name.split('_')[2]) in [56])

    @staticmethod
    def get_model_from_name(model_name, initializers: List[Initializer], outputs=10):
        """The naming scheme for a ResNet is 'cifar_resnet_D[_W]'.
        D is the model depth (e.g. cifar_resnet56)

        """

        if not CIFAR_ResNet.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        depth = int(model_name.split('_')[2])
        if len(model_name.split('_')) == 3:
            width = 16
        else:
            width = int(model_name.split('_')[4])

        if (depth - 2) % 3 != 0:
            raise ValueError('Invalid CIFAR_ResNet depth: {}'.format(depth))
        num_blocks = (depth - 2) // 6

        model_arch = {
            56: [(width, num_blocks), (2 * width, num_blocks), (4 * width, num_blocks)],
        }

        return CIFAR_ResNet(model_arch[depth], initializers, outputs)
