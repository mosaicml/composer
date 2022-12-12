# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""CIFAR image classification dataset.

The CIFAR datasets are a collection of labeled 32x32 colour images. Please refer to the `CIFAR dataset
<https://www.cs.toronto.edu/~kriz/cifar.html>`_ for more details.
"""

import os
import textwrap
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer.core import DataSpec, MemoryFormat
from composer.datasets.ffcv_utils import write_ffcv_dataset
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import pil_image_collate
from composer.utils import MissingConditionalImportError, dist

__all__ = [
    'build_cifar10_dataloader', 'build_ffcv_cifar10_dataloader', 'build_streaming_cifar10_dataloader',
    'build_synthetic_cifar10_dataloader'
]

CIFAR10_CHANNEL_MEAN = 0.4914, 0.4822, 0.4465
CIFAR10_CHANNEL_STD = 0.247, 0.243, 0.261


def build_cifar10_dataloader(
    datadir: str,
    global_batch_size: int,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Any,
) -> DataSpec:
    """Builds a CIFAR-10 dataloader with default transforms.

    Args:
        datadir (str): Path to the data directory
        global_batch_size (int): Global batch size
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
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD),
        ])

    with dist.run_local_rank_zero_first():
        dataset = datasets.CIFAR10(
            datadir,
            train=is_train,
            download=dist.get_local_rank() == 0 and download,
            transform=transform,
        )

    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            **dataloader_kwargs,
        ),)


def build_ffcv_cifar10_dataloader(
    global_batch_size: int,
    is_train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    num_workers: int = 8,
    ffcv_dir: str = '/tmp',
    ffcv_dest: str = 'cifar_train.ffcv',
    ffcv_write_dataset: Union[str, bool] = False,
    datadir: Union[str, None] = None,
) -> DataSpec:
    """Builds an FFCV CIFAR10 dataloader.

    Args:
        global_batch_size (int): Global batch size.
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
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
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

    return DataSpec(
        ffcv.Loader(
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
        ),)


def build_synthetic_cifar10_dataloader(
    global_batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    num_unique_samples: int = 100,
    device: str = 'cpu',
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS_FORMAT,
    **dataloader_kwargs: Any,
) -> DataSpec:
    """Builds a synthetic CIFAR-10 dataset for debugging or profiling.

    Args:
        global_batch_size (int): Global batch size
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
        total_dataset_size=50_000 if is_train else 10_000,
        data_shape=[3, 32, 32],
        num_classes=10,
        num_unique_samples_to_create=num_unique_samples,
        device=device,
        memory_format=memory_format,
    )
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            **dataloader_kwargs,
        ),)


def build_streaming_cifar10_dataloader(
    global_batch_size: int,
    remote: str,
    *,
    local: str = '/tmp/mds-cache/mds-cifar10',
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    predownload: Optional[int] = 100_000,
    keep_zip: Optional[bool] = None,
    download_retry: int = 2,
    download_timeout: float = 60,
    validate_hash: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs: Dict[str, Any],
) -> DataSpec:
    """Builds a streaming CIFAR10 dataset

    Args:
        global_batch_size (int): Global batch size.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str, optional): Local filesystem directory where dataset is cached during operation.
            Defaults to ``'/tmp/mds-cache/mds-imagenet1k/```.
        split (str): Which split of the dataset to use. Either ['train', 'val']. Default:
            ``'train```.
        drop_last (bool, optional): whether to drop last samples. Default: ``True``.
        shuffle (bool, optional): whether to shuffle dataset. Defaults to ``True``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()

    try:
        from streaming.vision import StreamingCIFAR10
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e

    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_CHANNEL_MEAN, CIFAR10_CHANNEL_STD),
        ])

    dataset = StreamingCIFAR10(
        local=local,
        remote=remote,
        split=split,
        shuffle=shuffle,
        transform=transform,
        predownload=predownload,
        keep_zip=keep_zip,
        download_retry=download_retry,
        download_timeout=download_timeout,
        validate_hash=validate_hash,
        shuffle_seed=shuffle_seed,
        num_canonical_nodes=num_canonical_nodes,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=pil_image_collate,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return DataSpec(dataloader=dataloader)
