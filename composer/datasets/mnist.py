# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer.core import MemoryFormat
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist


def build_mnist_dataloader(
    datadir: str,
    global_batch_size: int,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Builds an MNIST dataloader.

    Args:
        datadir (str): Path to the data directory
        global_batch_size (int): Global batch size.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        download (bool, optional): Whether to download the dataset, if needed. Default:
            ``True``.
        drop_last (bool): Drop remainder samples. Default: ``True``.
        shuffle (bool): Shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()

    transform = transforms.Compose([transforms.ToTensor()])

    with dist.run_local_rank_zero_first():
        dataset = datasets.MNIST(
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


def build_synthetic_mnist_dataloader(
    global_batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    num_unique_samples: int = 100,
    device: str = 'cpu',
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS_FORMAT,
    **dataloader_kwargs: Any,
) -> DataLoader:
    """Builds a synthetic MNIST dataset.

    Args:
        global_batch_size (int): Global batch size.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): Drop remainder samples. Default: ``True``.
        shuffle (bool): Shuffle the dataset. Default: ``True``.
        num_unique_samples (int): number of unique samples in synthetic dataset. Default: ``100``.
        device (str): device with which to load the dataset. Default: ``cpu``.
        memory_format (:class:`composer.core.MemoryFormat`): memory format of the tensors. Default: ``CONTIGUOUS_FORMAT``.
        **dataloader_kwargs (Any): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    dataset = SyntheticBatchPairDataset(
        total_dataset_size=60_000 if is_train else 10_000,
        data_shape=[1, 28, 28],
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
