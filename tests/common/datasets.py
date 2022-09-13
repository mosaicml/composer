# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (5, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class RandomImageDataset(VisionDataset):
    """ Image Classification dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (32, 32, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (3, 32, 32), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        if is_PIL:  # PIL expects HWC
            shape = (shape[1], shape[2], shape[0])
        self.shape = shape
        self.num_classes = num_classes

        self.size = size
        self.x = None
        self.y = None

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype('uint8')
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y
