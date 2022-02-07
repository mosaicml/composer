# Copyright 2021 MosaicML. All Rights Reserved.

"""
Contains several commonly used objects (models, dataloaders) that are
shared across the test suite.
"""

from typing import Sequence

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from composer.models import ComposerClassifier
from composer.trainer.devices import DeviceCPU, DeviceGPU


# decorators for common parameterizations and marks
def device(*args):

    parameters = []
    for arg in args:
        if arg == 'cpu':
            parameters += [pytest.param(DeviceCPU(), id="cpu")]
        elif arg == 'gpu':
            parameters += [pytest.param(DeviceGPU(), id="gpu", marks=pytest.mark.gpu)]
        else:
            raise ValueError(f'arguments to @device must be cpu, gpu, got {arg}')

    def decorator(test):
        if not parameters:
            return test
        return pytest.mark.parametrize("device", parameters)(test)

    return decorator


class SimpleModel(ComposerClassifier):
    """Small classification model with 10 input features and 2 classes.

    Args:
        num_features (int): number of input features (default: 10)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 5, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, 5)
        fc2 = torch.nn.Linear(5, num_classes)

        net = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)

        # fc1 and fc2 are bound to the model class
        # for access during several surgery tests
        self.fc1 = fc1
        self.fc2 = fc2


class SimpleConvModel(ComposerClassifier):
    """Small convolutional classifer.

    Args:
        num_channels (int): number of input channels (default: 32)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_channels: int = 32, num_classes: int = 2) -> None:

        self.num_classes = num_classes
        self.num_channels = num_channels

        conv_args = dict(kernel_size=(3, 3), padding=1)
        conv1 = torch.nn.Conv2d(in_channels=num_channels, out_channels=8, **conv_args)
        conv2 = torch.nn.Conv2d(in_channels=8, out_channels=4, **conv_args)
        pool = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()
        fc1 = torch.nn.Linear(4, 16)
        fc2 = torch.nn.Linear(16, num_classes)

        net = torch.nn.Sequential(
            conv1,
            conv2,
            pool,
            flatten,
            fc1,
            fc2,
        )
        super().__init__(module=net)

        # bind these to class for access during
        # surgery tests
        self.conv1 = conv1
        self.conv2 = conv2


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: 5)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 100)
    """

    def __init__(self, shape: Sequence[int] = (5,), size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return (self.x[index], self.y[index])


class RandomImageDataset(VisionDataset):
    """ Image Classification dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (64, 64, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 100)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (64, 64, 3), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype("uint8")
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y
