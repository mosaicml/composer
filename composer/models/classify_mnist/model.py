# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A simple convolutional neural network extending :class:`.ComposerClassifier`."""

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from composer.models.initializers import Initializer
from composer.models.tasks import ComposerClassifier

__all__ = ['Model', 'mnist_model']


class Model(nn.Module):
    """Toy convolutional neural network architecture in pytorch for MNIST."""

    def __init__(self, initializers: Sequence[Union[str, Initializer]], num_classes: int = 10):
        super().__init__()

        self.num_classes = num_classes

        for initializer in initializers:
            initializer = Initializer(initializer)
            self.apply(initializer.get_initializer())

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, (4, 4))
        out = torch.flatten(out, 1, -1)
        out = self.fc1(out)
        out = F.relu(out)
        return self.fc2(out)


def mnist_model(num_classes: int = 10, initializers: Optional[List[Initializer]] = None):
    """Helper function to create a :class:`.ComposerClassifier` with a simple convolutional neural network.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``
        initializers (List[Initializer], optional): list of Initializers
            for the model. ``None`` for no initialization. Default: ``None``

    Returns:
        ComposerModel: instance of :class:`.ComposerClassifier` with a simple MNIST model.

    Example:

    .. testcode::

        from composer.models import mnist_model

        model = mnist_model()
    """

    if initializers is None:
        initializers = []

    model = Model(initializers, num_classes)
    composer_model = ComposerClassifier(module=model)
    return composer_model
