# Copyright 2021 MosaicML. All Rights Reserved.

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from composer.models.base import ComposerClassifier
from composer.models.model_hparams import Initializer


class Model(nn.Module):
    """Toy classifier for MNIST.

    Should not be used to evaluate any method.
    """

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


class MNIST_Classifier(ComposerClassifier):
    """A simple convolutional neural network.

    :class:`~composer.models.classify_mnist.model.MNIST_Classifier` is a simple example
    convolutional neural network which can be used to classify MNIST data.

    Example:

    .. testcode::

        from composer.models import MNIST_Classifier

        model = MNIST_Classifier()

    Args:
        num_classes (int): The number of classes. Needed for classification tasks. Default = 10
        initializers (List[Initializer], optional): list of Initializers
            for the model. ``None`` for no initialization. (default: ``None``)
    """

    def __init__(
        self,
        num_classes: int = 10,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:
        if initializers is None:
            initializers = []

        model = Model(initializers, num_classes)
        super().__init__(module=model)
