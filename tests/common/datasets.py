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


class RandomSegmentationDataset(VisionDataset):
    """ Image Segmentation dataset with values drawn from a normal distribution
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
            mask_shape = self.shape[:2] if self.is_PIL else self.shape[1:]
            self.y = torch.randint(0, self.num_classes, size=(self.size, *mask_shape))
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


class RandomTextClassificationDataset(torch.utils.data.Dataset):
    """ Text classification dataset with values (just input token ids) drawn uniformly
    Args:
        vocab_size (int): vocab size to use (default: 10)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        sequence_length (int): sequence length to use, all sequences will be of this length with no padding (default: 8)
        use_keys: (bool): whether to return the item in a dictionary with keys for input and output
    """

    def __init__(self,
                 size: int = 100,
                 vocab_size: int = 10,
                 sequence_length: int = 8,
                 num_classes: int = 2,
                 use_keys: bool = False):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.use_keys = use_keys

        self.input_key = 'input_ids'
        self.label_key = 'labels'

        self.size = size
        self.x = None
        self.y = None

        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if self.x is None:
            self.x = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))
        if self.y is None:
            self.y = torch.randint(low=0, high=self.num_classes, size=(self.size,))

        x = self.x[index]
        y = self.y[index]

        if self.use_keys:
            return {'input_ids': x, 'labels': y}
        else:
            return x, y
