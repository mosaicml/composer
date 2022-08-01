# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer to the `CIFAR dataset
<https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

import os
from typing import Any, Optional

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

from composer.datasets.streaming import StreamingDataset

__all__ = ['StreamingCIFAR10']


class StreamingCIFAR10(StreamingDataset, VisionDataset):
    """
    Implementation of the CIFAR10 dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def decode_image(self, data: bytes) -> Image.Image:
        """Decode the sample image.

        Args:
            data (bytes): The raw bytes.

        Returns:
            Image: PIL image encoded by the bytes.
        """
        arr = np.frombuffer(data, np.uint8)
        arr = arr.reshape(32, 32, 3)
        return Image.fromarray(arr)

    def decode_class(self, data: bytes) -> np.int64:
        """Decode the sample class.

        Args:
            data (bytes): The raw bytes.

        Returns:
            np.int64: The class encoded by the bytes.
        """
        return np.frombuffer(data, np.int64)[0]

    def __init__(self, remote: str, local: str, split: str, shuffle: bool, batch_size: Optional[int] = None):
        # Build StreamingDataset
        decoders = {
            'x': self.decode_image,
            'y': self.decode_class,
        }
        super().__init__(remote=os.path.join(remote, split),
                         local=os.path.join(local, split),
                         shuffle=shuffle,
                         decoders=decoders,
                         batch_size=batch_size)

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")

        # Define custom transforms
        channel_means = 0.4914, 0.4822, 0.4465
        channel_stds = 0.247, 0.243, 0.261
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomCrop(32, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(channel_means, channel_stds),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(channel_means, channel_stds),
            ])
        VisionDataset.__init__(self, root=local, transform=transform)

    def __getitem__(self, idx: int) -> Any:
        """Get the decoded and transformed (image, class) pair by ID.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: Pair of (x, y) for this sample.
        """
        obj = super().__getitem__(idx)
        x = obj['x']
        assert self.transform is not None, 'transform set in __init__'
        x = self.transform(x)
        y = obj['y']
        return x, y
