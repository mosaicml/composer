"""
Contains several commonly used objects (models, dataloaders)
that are shared across the test suite.
"""
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from composer.models import ComposerClassifier


class SimpleModel(ComposerClassifier):
    """ Small classification model with 10 input features and 2 classes.

    Args:
        num_features (int): number of input features (default: 10)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 10, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        # fc1 and fc2 are bound to the model class
        # for access during several surgery tests
        self.fc1 = torch.nn.Linear(num_features, 5)
        self.fc2 = torch.nn.Linear(5, num_classes)

        self.net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=self.net)


class SimpleConvModel(ComposerClassifier):
    """ Small convolutional classifer

    Args:
        num_channels (int): number of input channels (default: 8)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_channels: int = 32, num_classes: int = 2) -> None:

        self.num_classes = num_classes
        self.num_channels = num_channels

        # fc1 and fc2 are bound to the model class
        # for access during several surgery tests
        conv_args = dict(kernel_size=(3, 3), padding=1)
        self.conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=8, **conv_args)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, **conv_args)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(4, num_classes)

        self.net = torch.nn.Sequential(
            self.conv1,
            self.conv2,
            self.pool,
            self.flatten,
            self.fc,
        )
        super().__init__(module=self.net)


class RandomClassificationDataset(Dataset):
    """ Classification dataset drawn from a normal distribution

    Args:
        shape (Sequence[int]): shape of features
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 100)
    """

    def __init__(self, shape: Sequence[int], size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return (self.x[index], self.y[index])


class RandomImageDataset(RandomClassificationDataset):
    """ Image Classification dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (64, 64, 32)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 100)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self,
                 shape: Sequence[int] = (64, 64, 32),
                 size: int = 100,
                 num_classes: int = 2,
                 is_PIL: bool = False):
        self.is_PIL = is_PIL
        super().__init__(shape=shape)

    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype("uint8")
            x = Image.fromarray(x)

        return (x, y)
