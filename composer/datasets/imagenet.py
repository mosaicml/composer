# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ImageNet classfication dataset.

The most widely used dataset for Image Classification algorithms. Please refer to the `ImageNet 2012 Classification
Dataset <http://image-net.org/>`_ for more details. Also includes streaming dataset versions based on the `WebDatasets
<https://github.com/webdataset/webdataset>`_.
"""

import os
import textwrap
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import ImageFolder

from composer.core import DataSpec
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.ffcv_utils import ffcv_monkey_patches, write_ffcv_dataset
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.streaming import StreamingImageClassDataset
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import dist

# ImageNet normalization values from torchvision: https://pytorch.org/vision/stable/models.html
IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)

__all__ = [
    "ImagenetDatasetHparams", "Imagenet1kWebDatasetHparams", "TinyImagenet200WebDatasetHparams", "StreamingImageNet1k",
    "StreamingImageNet1kHparams"
]


@dataclass
class ImagenetDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet dataset for image classification.

    Args:
        resize_size (int, optional): The resize size to use. Use ``-1`` to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        use_ffcv (bool): Whether to use FFCV dataloaders. Default: ``False``.
        ffcv_dir (str): A directory containing train/val <file>.ffcv files. If these files don't exist and
            ``ffcv_write_dataset`` is ``True``, train/val <file>.ffcv files will be created in this dir. Default: ``"/tmp"``.
        ffcv_dest (str): <file>.ffcv file that has dataset samples. Default: ``"imagenet_train.ffcv"``.
        ffcv_write_dataset (std): Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist. Default:
        ``False``.
    """
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)
    use_ffcv: bool = hp.optional("whether to use ffcv for faster dataloading", default=False)
    ffcv_dir: str = hp.optional(
        "A directory containing train/val <file>.ffcv files. If these files don't exist and ffcv_write_dataset is true, train/val <file>.ffcv files will be created in this dir.",
        default="/tmp")
    ffcv_dest: str = hp.optional("<file>.ffcv file that has dataset samples", default="imagenet_train.ffcv")
    ffcv_write_dataset: bool = hp.optional("Whether to create dataset in FFCV format (<file>.ffcv) if it doesn't exist",
                                           default=False)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:

        if self.use_synthetic:
            total_dataset_size = 1_281_167 if self.is_train else 50_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, self.crop_size, self.crop_size],
                num_classes=1000,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
            collate_fn = None
            device_transform_fn = None
        elif self.use_ffcv:
            try:
                import ffcv  # type: ignore
                from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder  # type: ignore
                from ffcv.fields.decoders import CenterCropRGBImageDecoder, IntDecoder  # type: ignore
                from ffcv.pipeline.operation import Operation  # type: ignore
            except ImportError:
                raise ImportError(
                    textwrap.dedent("""\
                    Composer was installed without ffcv support.
                    To use ffcv with Composer, please install ffcv in your environment."""))

            if self.is_train:
                split = "train"
            else:
                split = "val"
            dataset_filepath = os.path.join(self.ffcv_dir, self.ffcv_dest)
            # always create if ffcv_write_dataset is true
            if self.ffcv_write_dataset:
                if dist.get_local_rank() == 0:
                    if self.datadir is None:
                        raise ValueError(
                            "datadir is required if use_synthetic is False and ffcv_write_dataset is True.")
                    ds = ImageFolder(os.path.join(self.datadir, split))
                    write_ffcv_dataset(dataset=ds,
                                       write_path=dataset_filepath,
                                       max_resolution=500,
                                       num_workers=dataloader_hparams.num_workers,
                                       compress_probability=0.50,
                                       jpeg_quality=90)
                # Wait for the local rank 0 to be done creating the dataset in ffcv format.
                dist.barrier()

            this_device = torch.device(f'cuda:{dist.get_local_rank()}')
            label_pipeline: List[Operation] = [
                IntDecoder(),
                ffcv.transforms.ToTensor(),
                ffcv.transforms.Squeeze(),
                ffcv.transforms.ToDevice(this_device, non_blocking=True)
            ]
            image_pipeline: List[Operation] = []
            if self.is_train:
                image_pipeline.extend([
                    RandomResizedCropRGBImageDecoder((self.crop_size, self.crop_size)),
                    ffcv.transforms.RandomHorizontalFlip()
                ])
                dtype = np.float16
            else:
                ratio = self.crop_size / self.resize_size if self.resize_size > 0 else 1.0
                image_pipeline.extend([CenterCropRGBImageDecoder((self.crop_size, self.crop_size), ratio=ratio)])
                dtype = np.float32
            # Common transforms for train and test
            image_pipeline.extend([
                ffcv.transforms.ToTensor(),
                ffcv.transforms.ToDevice(this_device, non_blocking=True),
                ffcv.transforms.ToTorchImage(),
                ffcv.transforms.NormalizeImage(np.array(IMAGENET_CHANNEL_MEAN), np.array(IMAGENET_CHANNEL_STD), dtype),
            ])

            is_distributed = dist.get_world_size() > 1

            ffcv_monkey_patches()
            ordering = ffcv.loader.OrderOption.RANDOM if self.is_train else ffcv.loader.OrderOption.SEQUENTIAL

            return ffcv.Loader(
                dataset_filepath,
                batch_size=batch_size,
                num_workers=dataloader_hparams.num_workers,
                order=ordering,
                distributed=is_distributed,
                pipelines={
                    'image': image_pipeline,
                    'label': label_pipeline
                },
                batches_ahead=dataloader_hparams.prefetch_factor,
                drop_last=self.drop_last,
            )

        else:

            if self.is_train:
                # include fixed-size resize before RandomResizedCrop in training only
                # if requested (by specifying a size > 0)
                train_resize_size = self.resize_size
                train_transforms: List[torch.nn.Module] = []
                if train_resize_size > 0:
                    train_transforms.append(transforms.Resize(train_resize_size))
                # always include RandomResizedCrop and RandomHorizontalFlip
                train_transforms += [
                    transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                    transforms.RandomHorizontalFlip()
                ]
                transformation = transforms.Compose(train_transforms)
                split = "train"
            else:
                transformation = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                ])
                split = "val"

            device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
            collate_fn = pil_image_collate

            if self.datadir is None:
                raise ValueError("datadir must be specified if self.synthetic is False")
            dataset = ImageFolder(os.path.join(self.datadir, split), transformation)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)


class StreamingImageNet1k(StreamingImageClassDataset):
    """
    Implementation of the ImageNet1k dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        resize_size (int, optional): The resize size to use. Use -1 to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 resize_size: int = -1,
                 crop_size: int = 224,
                 batch_size: Optional[int] = None):

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if crop_size <= 0:
            raise ValueError(f"crop_size must be positive.")

        # Define custom transforms
        if split == "train":
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

        # Build StreamingDataset
        super().__init__(remote=os.path.join(remote, split),
                         local=os.path.join(local, split),
                         shuffle=shuffle,
                         transform=transform,
                         batch_size=batch_size)


@dataclass
class StreamingImageNet1kHparams(DatasetHparams):
    """DatasetHparams for creating an instance of StreamingImageNet1k.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-imagenet1k/mds/1/```
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-imagenet1k/```
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
        resize_size (int, optional): The resize size to use. Use -1 to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-imagenet1k/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-imagenet1k/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')
    resize_size: int = hp.optional("Resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("Crop size", default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        dataset = StreamingImageNet1k(remote=self.remote,
                                      local=self.local,
                                      split=self.split,
                                      shuffle=self.shuffle,
                                      resize_size=self.resize_size,
                                      crop_size=self.crop_size,
                                      batch_size=batch_size)
        collate_fn = pil_image_collate
        device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)


@dataclass
class TinyImagenet200WebDatasetHparams(WebDatasetHparams):
    """Defines an instance of the TinyImagenet-200 WebDataset for image classification.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-tinyimagenet200'``.
        name (str): Key used to determine where dataset is cached on local filesystem. Default: ``'tinyimagenet200'``.
        n_train_samples (int): Number of training samples. Default: ``100000``.
        n_val_samples (int): Number of validation samples. Default: ``10000``.
        height (int): Sample image height in pixels. Default: ``64``.
        width (int): Sample image width in pixels. Default: ``64``.
        n_classes (int): Number of output classes. Default: ``200``.
        channel_means (list of float): Channel means for normalization. Default: ``(0.485, 0.456, 0.406)``.
        channel_stds (list of float): Channel stds for normalization. Default: ``(0.229, 0.224, 0.225)``.
    """

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-tinyimagenet200')
    name: str = hp.optional('WebDataset local cache name', default='tinyimagenet200')

    n_train_samples: int = hp.optional('Number of samples in training split', default=100_000)
    n_val_samples: int = hp.optional('Number of samples in validation split', default=10_000)
    height: int = hp.optional('Image height', default=64)
    width: int = hp.optional('Image width', default=64)
    n_classes: int = hp.optional('Number of output classes', default=200)
    channel_means: List[float] = hp.optional('Mean per image channel', default=(0.485, 0.456, 0.406))
    channel_stds: List[float] = hp.optional('Std per image channel', default=(0.229, 0.224, 0.225))

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        from composer.datasets.webdataset_utils import load_webdataset

        if self.is_train:
            split = 'train'
            transform = transforms.Compose([
                transforms.RandomCrop((self.height, self.width), (self.height // 8, self.width // 8)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.channel_means, self.channel_stds),
            ])
        else:
            split = 'val'
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.channel_means, self.channel_stds),
            ])
        preprocess = lambda dataset: dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
        dataset = load_webdataset(self.remote, self.name, split, self.webdataset_cache_dir,
                                  self.webdataset_cache_verbose, self.shuffle, self.shuffle_buffer, preprocess,
                                  dist.get_world_size(), dataloader_hparams.num_workers, batch_size, self.drop_last)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            drop_last=self.drop_last,
            collate_fn=None,
        ),
                        device_transforms=None)


@dataclass
class Imagenet1kWebDatasetHparams(WebDatasetHparams):
    """Defines an instance of the ImageNet-1k WebDataset for image classification.

    Args:
        remote (str): S3 bucket or root directory where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-imagenet1k'``.
        name (str): Key used to determine where dataset is cached on local filesystem. Default: ``'imagenet1k'``.
        resize_size (int, optional): The resize size to use. Use -1 to not resize. Default: ``-1``.
        crop size (int): The crop size to use. Default: ``224``.
    """

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-imagenet1k')
    name: str = hp.optional('WebDataset local cache name', default='imagenet1k')
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        from composer.datasets.webdataset_utils import load_webdataset

        if self.is_train:
            # include fixed-size resize before RandomResizedCrop in training only
            # if requested (by specifying a size > 0)
            train_resize_size = self.resize_size
            train_transforms: List[torch.nn.Module] = []
            if train_resize_size > 0:
                train_transforms.append(transforms.Resize(train_resize_size))
            # always include RandomResizedCrop and RandomHorizontalFlip
            train_transforms += [
                transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
                transforms.RandomHorizontalFlip()
            ]
            transform = transforms.Compose(train_transforms)
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
            ])
        split = 'train' if self.is_train else 'val'
        preprocess = lambda dataset: dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
        dataset = load_webdataset(self.remote, self.name, split, self.webdataset_cache_dir,
                                  self.webdataset_cache_verbose, self.shuffle, self.shuffle_buffer, preprocess,
                                  dist.get_world_size(), dataloader_hparams.num_workers, batch_size, self.drop_last)
        collate_fn = pil_image_collate
        device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        ),
                        device_transforms=device_transform_fn)
