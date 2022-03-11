# Copyright 2021 MosaicML. All Rights Reserved.

"""A ResNet-9 model extending :class:`.ComposerClassifier` and ResNet-9 architecture."""

import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from composer.models.base import ComposerClassifier

__all__ = ["ResNet9", "CIFAR10_ResNet9"]


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
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        out = self.body(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = self.fc(out)
        return out


class CIFAR10_ResNet9(ComposerClassifier):
    """A ResNet-9 model extending :class:`.ComposerClassifier`.

    See `myrtle.ai blog <https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/>`_ for more details.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.

    Example:

    .. testcode::

        from composer.models import CIFAR10_ResNet9

        model = CIFAR10_ResNet9()  # creates a resnet9 for cifar image classification
    """

    def __init__(self, num_classes: int = 10) -> None:
        model = ResNet9(num_classes)
        super().__init__(module=model)
