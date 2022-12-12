# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classification streaming dataset.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet 2012 Classification
Dataset <http://image-net.org/>`_ for more details.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from composer.core import DataSpec, MemoryFormat
from composer.datasets.ffcv_utils import ffcv_monkey_patches, write_ffcv_dataset
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import MissingConditionalImportError, dist

__all__ = [
    'build_imagenet_dataloader',
    'build_streaming_imagenet1k_dataloader',
    'build_synthetic_imagenet_dataloader',
    'write_ffcv_imagenet',
    'build_ffcv_imagenet_dataloader',
]

IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def build_imagenet_dataloader(
    datadir: str,
    global_batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    **dataloader_kwargs: Dict[str, Any],
) -> DataSpec:
    """Builds an ImageNet dataloader.

    Args:
        datadir (str): path to location of dataset.
        global_batch_size (int): Global batch size.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    if is_train:
        # include fixed-size resize before RandomResizedCrop in training only
        # if requested (by specifying a size > 0)
        train_transforms: List[torch.nn.Module] = []

        if resize_size > 0:
            train_transforms.append(transforms.Resize(resize_size))

        train_transforms += [
            transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip()
        ]
        transformation = transforms.Compose(train_transforms)
        split = 'train'
    else:
        val_transforms: List[torch.nn.Module] = []
        if resize_size > 0:
            val_transforms.append(transforms.Resize(resize_size))
        val_transforms.append(transforms.CenterCrop(crop_size))
        transformation = transforms.Compose(val_transforms)
        split = 'val'

    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)

    dataset = ImageFolder(os.path.join(datadir, split), transformation)
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataSpec(
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )


def build_synthetic_imagenet_dataloader(
    global_batch_size: int,
    num_unique_samples: int = 100,
    device: str = 'cpu',
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS_FORMAT,
    is_train: bool = True,
    crop_size: int = 224,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs: Dict[str, Any],
) -> DataSpec:
    """Builds a synthetic ImageNet dataloader.

    Args:
        global_batch_size (int): Global batch size.
        num_unique_samples (int): number of unique samples in synthetic dataset. Default: ``100``.
        device (str): device with which to load the dataset. Default: ``cpu``.
        memory_format (:class:`composer.core.MemoryFormat`): memory format of the tensors. Default: ``CONTIGUOUS_FORMAT``.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        crop size (int): The crop size to use. Default: ``224``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    total_dataset_size = 1_281_167 if is_train else 50_000
    dataset = SyntheticBatchPairDataset(
        total_dataset_size=total_dataset_size,
        data_shape=[3, crop_size, crop_size],
        num_classes=1000,
        num_unique_samples_to_create=num_unique_samples,
        device=device,
        memory_format=memory_format,
    )

    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataSpec(
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=pil_image_collate,
            **dataloader_kwargs,
        ),)


def write_ffcv_imagenet(
    datadir: str,
    savedir: str = '/tmp/imagenet_train.ffcv',
    split: str = 'train',
    num_workers: int = 8,
):
    """Converts an ImageNet dataset to FFCV format.

        datadir (str): Path of ImageNet dataset, in ImageFolder format.
        savedir (str): Path to save the FFCV dataset. Default: ``/tmp/imagenet_train.ffcv``.
        split (str): 'train' or 'val'. Default: ``train``.
        num_workers (int): Number of workers to use for conversion. Default: ``8``.
    """

    if dist.get_local_rank() == 0:
        ds = ImageFolder(os.path.join(datadir, split))
        write_ffcv_dataset(dataset=ds,
                           write_path=savedir,
                           max_resolution=500,
                           num_workers=num_workers,
                           compress_probability=0.50,
                           jpeg_quality=90)

    # wait for rank 0 to finish conversion
    dist.barrier()


def build_ffcv_imagenet_dataloader(
    datadir: str,
    global_batch_size: int,
    is_train: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    cpu_only: bool = False,
    drop_last: bool = True,
    prefetch_factor: int = 2,
    num_workers: int = 8,
):
    """Builds an FFCV ImageNet dataloader.

    Args:
        datadir (str): path to location of dataset.
        global_batch_size (int): Global batch size.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        cpu_only (int): Only perform transforms on 'cpu'. Default: ``False``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        prefetch_factor (int): Number of batches to prefect. Default: ``2``.
        num_workers (int): Number of workers. Default: ``8``.
    """
    try:
        import ffcv
        from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder, RandomResizedCropRGBImageDecoder
        from ffcv.pipeline.operation import Operation
    except ImportError:
        raise ImportError('Composer was installed without ffcv support.'
                          'To use ffcv with Composer, please install ffcv.')
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    device = torch.device(f'cuda:{dist.get_local_rank()}')
    label_pipeline: List[Operation] = [
        IntDecoder(),
        ffcv.transforms.ToTensor(),
        ffcv.transforms.Squeeze(),
        ffcv.transforms.ToDevice(device, non_blocking=True)
    ]
    image_pipeline: List[Operation] = []
    if is_train:
        image_pipeline.extend(
            [RandomResizedCropRGBImageDecoder((crop_size, crop_size)),
             ffcv.transforms.RandomHorizontalFlip()])
        dtype = np.float16
    else:
        ratio = crop_size / resize_size if resize_size > 0 else 1.0
        image_pipeline.extend([CenterCropRGBImageDecoder((crop_size, crop_size), ratio=ratio)])
        dtype = np.float32

    # Common transforms for train and test
    if cpu_only:
        image_pipeline.extend([
            ffcv.transforms.NormalizeImage(np.array(IMAGENET_CHANNEL_MEAN), np.array(IMAGENET_CHANNEL_STD), dtype),
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToTorchImage(),
        ])
    else:
        image_pipeline.extend([
            ffcv.transforms.ToTensor(),
            ffcv.transforms.ToDevice(device, non_blocking=True),
            ffcv.transforms.ToTorchImage(),
            ffcv.transforms.NormalizeImage(np.array(IMAGENET_CHANNEL_MEAN), np.array(IMAGENET_CHANNEL_STD), dtype),
        ])

    is_distributed = dist.get_world_size() > 1

    ffcv_monkey_patches()
    ordering = ffcv.loader.OrderOption.RANDOM if is_train else ffcv.loader.OrderOption.SEQUENTIAL

    return ffcv.Loader(
        datadir,
        batch_size=batch_size,
        num_workers=num_workers,
        order=ordering,
        distributed=is_distributed,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        },
        batches_ahead=prefetch_factor,
        drop_last=drop_last,
    )


def build_streaming_imagenet1k_dataloader(
    global_batch_size: int,
    remote: str,
    *,
    local: str = '/tmp/mds-cache/mds-imagenet1k',
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    predownload: Optional[int] = 100_000,
    keep_zip: Optional[bool] = None,
    download_retry: int = 2,
    download_timeout: float = 60,
    validate_hash: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs: Dict[str, Any],
) -> DataSpec:
    """Builds an imagenet1k streaming dataset

    Args:
        global_batch_size (int): Global batch size.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str, optional): Local filesystem directory where dataset is cached during operation.
            Defaults to ``'/tmp/mds-cache/mds-imagenet1k/```.
        split (str): Which split of the dataset to use. Either ['train', 'val']. Default:
            ``'train```.
        drop_last (bool, optional): whether to drop last samples. Default: ``True``.
        shuffle (bool, optional): whether to shuffle dataset. Defaults to ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
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
        from streaming.vision import StreamingImageNet
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e

    transform = []
    if split == 'train':
        # include fixed-size resize before RandomResizedCrop in training only
        # if requested (by specifying a size > 0)
        if resize_size > 0:
            transform.append(transforms.Resize(resize_size))
        # always include RandomResizedCrop and RandomHorizontalFlip
        transform += [
            transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip()
        ]
    else:
        if resize_size > 0:
            transform.append(transforms.Resize(resize_size))
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(lambda image: image.convert('RGB'))
    transform = transforms.Compose(transform)

    dataset = StreamingImageNet(
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
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
    return DataSpec(dataloader=dataloader, device_transforms=device_transform_fn)
