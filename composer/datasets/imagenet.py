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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder, VisionDataset

from composer.core.data_spec import DataSpec
from composer.core.types import MemoryFormat
from composer.datasets.ffcv_utils import ffcv_monkey_patches, write_ffcv_dataset
from composer.datasets.streaming import StreamingDataset
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist

__all__ = [
    'StreamingImageNet1k',
    'build_imagenet_dataloader',
    'build_synthetic_imagenet_dataloader',
    'write_ffcv_imagenet',
    'build_ffcv_imagenet_dataloader',
]

IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def build_imagenet_dataloader(
    datadir: str,
    batch_size: int,
    is_train: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
    resize_size: int = -1,
    crop_size: int = 224,
    **dataloader_kwargs,
) -> DataSpec:
    """Builds an ImageNet dataloader.

    Args:
        datadir (str): path to location of dataset.
        batch_size (int): Batch size per device.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
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
    batch_size: int,
    num_unique_samples: int = 100,
    device: str = 'cpu',
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS_FORMAT,
    is_train: bool = True,
    crop_size: int = 224,
    drop_last: bool = True,
    shuffle: bool = True,
    **dataloader_kwargs,
) -> DataSpec:
    """Builds a synthetic ImageNet dataloader.

    Args:
        batch_size (int): Batch size per device.
        num_unique_samples (int): number of unique samples in synthetic dataset. Default: ``100``.
        device (str): device with which to load the dataset. Default: ``cpu``.
        memory_format (MemoryFormat): memory format of the tensors. Default: ``CONTIGUOUS_FORMAT``.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
        crop size (int): The crop size to use. Default: ``224``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
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
    batch_size: int,
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
        batch_size (int): Batch size per device.
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
