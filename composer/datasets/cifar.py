# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer to the `CIFAR dataset
<https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

import os
import textwrap
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

from composer.core.types import MemoryFormat
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.streaming import StreamingDataset
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist

__all__ = [
    'build_cifar10_dataloader', 'build_ffcv_cifar10_dataloader', 'build_synthetic_cifar10_dataloader',
    'StreamingCIFAR10'
]


def build_cifar10_dataloader(
    datadir: str,
    batch_size: int,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Builds a CIFAR-10 dataloader with default transforms.

    Args:
        datadir (str): Path to the data directory
        batch_size (int): Batch size per device
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        download (bool, optional): Whether to download the dataset, if needed. Default:
            ``True``.
        drop_last (bool): Drop remainder samples. Default: ``True``.
        shuffle (bool): Shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    cifar10_mean = 0.4914, 0.4822, 0.4465
    cifar10_std = 0.247, 0.243, 0.261
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])

    with dist.run_local_rank_zero_first():
        dataset = datasets.CIFAR10(
            datadir,
            train=is_train,
            download=dist.get_local_rank() == 0 and download,
            transform=transform,
        )

    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        **dataloader_kwargs,
    )


def build_ffcv_cifar10_dataloader(
    batch_size: int,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    num_workers: int = 8,
    ffcv_dir: str = '/tmp',
    ffcv_dest: str = 'cifar_train.ffcv',
    ffcv_write_dataset: Union[str, bool] = False,
    datadir: Union[str, None] = None,
):
    """Builds an FFCV CIFAR10 dataloader.

    Args:
        batch_size (int): Batch size per device.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        download (bool, optional): Whether to download the dataset, if needed. Default:
            ``True``.
        drop_last (bool): Whether to drop last samples. Default: ``True``.
        prefetch_factor (int): Number of batches to prefect. Default: ``2``.
        ffcv_dir (str, optional): A directory containing train/val <file>.ffcv files. If
            these files don't exist and ``ffcv_write_dataset`` is ``True``, train/val
            <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str, optional): <file>.ffcv file that has dataset samples. Default: ``"cifar_train.ffcv"``.
        ffcv_write_dataset (str | bool, optional): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
        datadir (str | None, optional): Path to the non-FFCV data directory.
    """
    try:
        import ffcv
        from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
        from ffcv.pipeline.operation import Operation
    except ImportError:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without ffcv support.
            To use ffcv with Composer, please install ffcv in your environment."""))

    dataset_filepath = os.path.join(ffcv_dir, ffcv_dest)
    # always create if ffcv_write_dataset is true
    if ffcv_write_dataset:
        if dist.get_local_rank() == 0:
            if datadir is None:
                raise ValueError('datadir is required if use_synthetic is False and ffcv_write_dataset is True.')
            ds = datasets.CIFAR10(
                datadir,
                train=is_train,
                download=download,
            )

            write_ffcv_dataset(dataset=ds, write_path=dataset_filepath)

        # Wait for the local rank 0 to be done creating the dataset in ffcv format.
        dist.barrier()

    if not os.path.exists(dataset_filepath):
        raise ValueError(
            f'Dataset file containing samples not found at {dataset_filepath}. Use ffcv_dir flag to point to a dir containing {dataset_filepath}.'
        )

    # Please note that this mean/std is different from the mean/std used for regular PyTorch dataloader as
    # ToTensor does the normalization for PyTorch dataloaders.
    cifar10_mean_ffcv = [125.307, 122.961, 113.8575]
    cifar10_std_ffcv = [51.5865, 50.847, 51.255]
    label_pipeline: List[Operation] = [IntDecoder(), ffcv.transforms.ToTensor(), ffcv.transforms.Squeeze()]
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    if is_train:
        image_pipeline.extend([
            ffcv.transforms.RandomHorizontalFlip(),
            ffcv.transforms.RandomTranslate(padding=2, fill=tuple(map(int, cifar10_mean_ffcv))),
            ffcv.transforms.Cutout(4, tuple(map(int, cifar10_mean_ffcv))),
        ])
    # Common transforms for train and test
    image_pipeline.extend([
        ffcv.transforms.ToTensor(),
        ffcv.transforms.ToTorchImage(channels_last=False, convert_back_int16=False),
        ffcv.transforms.Convert(torch.float32),
        transforms.Normalize(cifar10_mean_ffcv, cifar10_std_ffcv),
    ])

    ordering = ffcv.loader.OrderOption.RANDOM if is_train else ffcv.loader.OrderOption.SEQUENTIAL

    return ffcv.Loader(
        dataset_filepath,
        batch_size=batch_size,
        num_workers=num_workers,
        order=ordering,
        distributed=False,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline,
        },
        batches_ahead=prefetch_factor,
        drop_last=drop_last,
    )


def build_synthetic_cifar10_dataloader(
    batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    num_unique_samples: int = 100,
    device: str = 'cpu',
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS_FORMAT,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Builds a synthetic CIFAR-10 dataset for debugging or profiling.

    Args:
        batch_size (int): Batch size per device
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): Drop remainder samples. Default: ``True``.
        shuffle (bool): Shuffle the dataset. Default: ``True``.
        num_unique_samples (int): number of unique samples in synthetic dataset. Default: ``100``.
        device (str): device with which to load the dataset. Default: ``cpu``.
        memory_format (MemoryFormat): memory format of the tensors. Default: ``CONTIGUOUS_FORMAT``.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    dataset = SyntheticBatchPairDataset(
        total_dataset_size=50_000 if is_train else 10_000,
        data_shape=[3, 32, 32],
        num_classes=10,
        num_unique_samples_to_create=num_unique_samples,
        device=device,
        memory_format=memory_format,
    )
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        **dataloader_kwargs,
    )


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
