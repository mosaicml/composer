# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification streaming dataset.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet 2012 Classification
Dataset <http://image-net.org/>`_ for more details.
"""

import os
from io import BytesIO
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

from composer.datasets.streaming import StreamingDataset

__all__ = ['StreamingImageNet1k']


class StreamingImageNet1k(StreamingDataset, VisionDataset):
    """
    Implementation of the ImageNet1k dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        resize_size (int, optional): The resize size to use. Use -1 to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def decode_image(self, data: bytes) -> Image.Image:
        """Decode the sample image.

        Args:
            data (bytes): The raw bytes.

        Returns:
            Image: PIL image encoded by the bytes.
        """
        return Image.open(BytesIO(data)).convert('RGB')

    def decode_class(self, data: bytes) -> np.int64:
        """Decode the sample class.

        Args:
            data (bytes): The raw bytes.

        Returns:
            np.int64: The class encoded by the bytes.
        """
        return np.frombuffer(data, np.int64)[0]

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 resize_size: int = -1,
                 crop_size: int = 224,
                 batch_size: Optional[int] = None):
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
        if crop_size <= 0:
            raise ValueError(f'crop_size must be positive.')

        # Define custom transforms
        if split == 'train':
            # include fixed-size resize before RandomResizedCrop in training only
            # if requested (by specifying a size > 0)
            train_transforms: List[torch.nn.Module] = []
            if resize_size > 0:
                train_transforms.append(transforms.Resize(resize_size))
            # always include RandomResizedCrop and RandomHorizontalFlip
            train_transforms += [
                transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                transforms.RandomHorizontalFlip(),
            ]
            transform = transforms.Compose(train_transforms)
        else:
            val_transforms: List[torch.nn.Module] = []
            if resize_size > 0:
                val_transforms.append(transforms.Resize(resize_size))
            val_transforms += [transforms.CenterCrop(crop_size)]
            transform = transforms.Compose(val_transforms)
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
