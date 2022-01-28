# Copyright 2021 MosaicML. All Rights Reserved.

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import ImageFolder
from webdataset import WebDataset

from composer.core.types import DataLoader, DataSpec
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist
from composer.utils.data import NormalizationFn, pil_image_collate


# ImageNet normalization values from torchvision: https://pytorch.org/vision/stable/models.html
IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


@dataclass
class ImagenetDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet dataset for image classification.

    Parameters:
        resize_size (int, optional): The resize size to use. Defaults to -1 to not resize.
        crop size (int): The crop size to use.
    """
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:

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


@dataclass
class WebTinyImagenet200DatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the TinyImagenet-200 WebDataset for image classification.

    Parameters:
        train_shards (int): Number of training split shards.
        val_shards (int): Number of validation split shards.
    """
    train_shards: int = hp.optional('Training split shards', default=128)
    val_shards: int = hp.optional('Validation split shards', default=16)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        tinyimagenet_mean = 0.485, 0.456, 0.406
        tinyimagenet_std = 0.229, 0.224, 0.225

        if self.is_train:
            split = 'train'
            size = 100_000
            n_shards = self.train_shards
        else:
            split = 'val'
            size = 10_000
            n_shards = self.val_shards
        size = size - size % n_shards

        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=size,
                data_shape=[3, 64, 64],
                num_classes=200,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
        else:
            if self.is_train:
                transform = transforms.Compose([
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(tinyimagenet_mean, tinyimagenet_std),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(tinyimagenet_mean, tinyimagenet_std),
                ])

            urls = ['/datasets/web_tinyimagenet200/%s_%05d.tar' % (split, i) for i in range(n_shards)]
            size_per_device = size // dist.get_world_size()
            dataset = WebDataset(urls, cache_dir=self.dataset_cache_dir,
                                 cache_verbose=self.dataset_cache_verbose)
            dataset = dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
            dataset = dataset.with_epoch(size_per_device).with_length(size_per_device)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)


@dataclass
class WebImagenet1KDatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet-1K dataset for image classification.

    Parameters:
        resize_size (int, optional): The resize size to use. Defaults to -1 to not resize.
        crop size (int): The crop size to use.
        train_shards (int): Number of training split shards.
        val_shards (int): Number of validation split shards.
    """
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)

    train_shards: int = hp.optional('Training split shards', default=1024)
    val_shards: int = hp.optional('Validation split shards', default=128)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        if self.is_train:
            split = 'train'
            size = 1_281_167
            n_shards = self.train_shards
        else:
            split = 'val'
            size = 50_000
            n_shards = self.val_shards
        size = size - size % n_shards

        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=size,
                data_shape=[3, self.crop_size, self.crop_size],
                num_classes=1000,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
            collate_fn = None
            device_transform_fn = None
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
                transform = transforms.Compose(train_transforms)
            else:
                transform = transforms.Compose([
                    transforms.Resize(self.resize_size),
                    transforms.CenterCrop(self.crop_size),
                ])

            urls = ['/datasets/web_imagenet1k/%s_%05d.tar' % (split, i) for i in range(n_shards)]
            size_per_device = size // dist.get_world_size()
            dataset = WebDataset(urls, cache_dir=self.dataset_cache_dir,
                                 cache_verbose=self.dataset_cache_verbose)
            dataset = dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
            dataset = dataset.with_epoch(size_per_device).with_length(size_per_device)

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
