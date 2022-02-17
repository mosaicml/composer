# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional

import torch.nn as nn
from torchvision.models.resnet import BasicBlock

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer


# adapted from https://raw.githubusercontent.com/matthias-wright/cifar10-resnet/master/model.py
# under the MIT license
class ResNet9(nn.Module):
    """A 9-layer residual network, excluding BatchNorms and activation functions, as described in this blog post:

    https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
    """

    def __init__(self, num_classes: int):
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
    """A ResNet-9 model extending :class:`ComposerClassifier`.

    See this blog post for details regarding the architecture:
    https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/

    Args:
        num_classes (int): The number of classes for the model.
        initializers (List[Initializer], optional): Initializers
            for the model. ``None`` for no initialization.
            (default: ``None``)
    """

    def __init__(
        self,
        num_classes: int,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        model = ResNet9(num_classes)
        super().__init__(module=model)
