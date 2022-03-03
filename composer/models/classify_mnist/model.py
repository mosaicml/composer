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

    def __init__(self, initializers: Sequence[Union[str, Initializer]], outputs: int):
        super().__init__()

        self.num_classes = outputs

        for initializer in initializers:
            initializer = Initializer(initializer)
            self.apply(initializer.get_initializer())

        self.conv1 = nn.Conv2d(1, 16, (3, 3), padding=0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=0)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 16, 32)
        self.fc2 = nn.Linear(32, outputs)

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
    """A simple convolutional neural network extending :class:`ComposerClassifier`.

    :class:`composer.models.MNIST_Classifier` is a simple example
    convolutional neural network which can be used to classify MNIST data.
    """

    def __init__(
        self,
        num_classes: int,
        initializers: Optional[List[Initializer]] = None,
    ) -> None:

        if initializers is None:
            initializers = []

        model = Model(initializers, num_classes)
        super().__init__(module=model)
