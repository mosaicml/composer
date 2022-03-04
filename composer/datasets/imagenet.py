# Copyright 2021 MosaicML. All Rights Reserved.

import os
from dataclasses import dataclass
from typing import List

import torch
import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import ImageFolder

from composer.core.types import DataLoader, DataSpec
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.datasets.webdataset import load_webdataset, size_webdataset
from composer.utils import dist

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
class TinyImagenet200WebDatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the TinyImagenet-200 WebDataset for image classification.

    Parameters:
        webdataset_s3_bucket (str): S3 bucket or root directory where dataset is stored.
        webdataset_name (str): Key used to determine where dataset is cached on local filesystem.
        n_train_samples (int): Number of training samples.
        n_val_samples (int): Number of validation samples.
        height (int): Sample image height in pixels.
        width (int): Sample image width in pixels.
        n_classes (int): Number of output classes.
        channel_means (list of float): Channel means for normalization.
        channel_stds (list of float): Channel stds for normalization.
    """

    webdataset_s3_bucket: str = hp.optional('WebDataset S3 bucket name',
                                            default='mosaicml-internal-dataset-tinyimagenet200')
    webdataset_name: str = hp.optional('WebDataset local cache name', default='tinyimagenet200')
    n_train_samples = 100_000
    n_val_samples = 10_000
    height = 64
    width = 64
    n_classes = 200
    channel_means = 0.485, 0.456, 0.406
    channel_stds = 0.229, 0.224, 0.225

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
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
        dataset, meta = load_webdataset(self.webdataset_s3_bucket, self.webdataset_name, split,
                                        self.webdataset_cache_dir, self.webdataset_cache_verbose)
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_per_worker)
        dataset = dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
        dataset = size_webdataset(dataset, meta['n_shards'], meta['samples_per_shard'], dist.get_world_size(),
                                  dataloader_hparams.num_workers, batch_size, self.drop_last)
        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)


@dataclass
class Imagenet1kWebDatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the ImageNet-1k dataset for image classification.

    Parameters:
        webdataset_s3_bucket (str): S3 bucket or root directory where dataset is stored.
        webdataset_name (str): Key used to determine where dataset is cached on local filesystem.
        resize_size (int, optional): The resize size to use. Defaults to -1 to not resize.
        crop size (int): The crop size to use.
    """

    webdataset_s3_bucket: str = hp.optional('WebDataset S3 bucket name', default='mosaicml-internal-dataset-imagenet1k')
    webdataset_name: str = hp.optional('WebDataset local cache name', default='imagenet1k')
    resize_size: int = hp.optional("resize size. Set to -1 to not resize", default=-1)
    crop_size: int = hp.optional("crop size", default=224)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
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
        dataset, meta = load_webdataset(self.webdataset_s3_bucket, self.webdataset_name, split,
                                        self.webdataset_cache_dir, self.webdataset_cache_verbose)
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_per_worker)
        dataset = dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
        dataset = size_webdataset(dataset, meta['n_shards'], meta['samples_per_shard'], dist.get_world_size(),
                                  dataloader_hparams.num_workers, batch_size, self.drop_last)
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
