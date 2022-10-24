# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Contains commonly used models that are shared across the test suite."""

import torch

from composer.models import ComposerClassifier


class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_features: int = 1, num_classes: int = 2) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, 5)
        fc2 = torch.nn.Linear(5, num_classes)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        super().__init__(module=net)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2


class SimpleConvModel(ComposerClassifier):
    """Small convolutional classifer.

    Args:
        num_channels (int): number of input channels (default: 3)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 2) -> None:

        self.num_classes = num_classes
        self.num_channels = num_channels

        conv_args = {'kernel_size': (3, 3), 'padding': 1, 'stride': 2}
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

        # bind these to class for access during surgery tests
        self.conv1 = conv1
        self.conv2 = conv2


class ConvModel(ComposerClassifier):
    """Convolutional network featuring strided convs, a batchnorm, max pooling, and average pooling."""

    def __init__(self):
        conv_args = {'kernel_size': (3, 3), 'padding': 1}
        conv1 = torch.nn.Conv2d(in_channels=32, out_channels=8, stride=2, bias=False, **conv_args)  # stride > 1
        conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, stride=2, bias=False,
                                **conv_args)  # stride > 1 but in_channels < 16
        conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, bias=False, **conv_args)  # stride = 1
        bn = torch.nn.BatchNorm2d(num_features=64)
        pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        pool2 = torch.nn.AdaptiveAvgPool2d(1)
        flatten = torch.nn.Flatten()
        linear1 = torch.nn.Linear(64, 48)
        linear2 = torch.nn.Linear(48, 10)

        net = torch.nn.Sequential(
            conv1,
            conv2,
            conv3,
            bn,
            pool1,
            pool2,
            flatten,
            linear1,
            linear2,
        )

        super().__init__(module=net)

        # bind these to class for access during surgery tests
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.bn = bn
        self.pool1 = pool1
        self.pool2 = pool2
        self.flatten = flatten
        self.linear1 = linear1
        self.linear2 = linear2
