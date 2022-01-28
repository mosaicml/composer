# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10
from webdataset import WebDataset

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils import dist


@dataclass
class CIFAR10DatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification.

    Parameters:
        download (bool): Whether to download the dataset, if needed.
    """
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        cifar10_mean = 0.4914, 0.4822, 0.4465
        cifar10_std = 0.247, 0.243, 0.261

        if self.use_synthetic:
            total_dataset_size = 50_000 if self.is_train else 10_000
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=total_dataset_size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        else:
            if self.datadir is None:
                raise ValueError("datadir is required if use_synthetic is False")

            if self.is_train:
                transformation = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                ])
            else:
                transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cifar10_mean, cifar10_std),
                ])

            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)


@dataclass
class WebCIFAR10DatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-10 WebDataset for image classification.

    Parameters:
        train_shards (int): Number of training split shards.
        val_shards (int): Number of validation split shards.
    """
    train_shards: int = hp.optional('Training split shards', default=16)
    val_shards: int = hp.optional('Validation split shards', default=8)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        cifar10_mean = 0.4914, 0.4822, 0.4465
        cifar10_std = 0.247, 0.243, 0.261

        if self.is_train:
            split = 'train'
            size = 50_000
            n_shards = self.train_shards
        else:
            split = 'val'
            size = 10_000
            n_shards = self.val_shards
        size = size - size % n_shards

        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=size,
                data_shape=[3, 32, 32],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
        else:
            if self.is_train:
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

            urls = ['/datasets/web_cifar10/%s_%05d.tar' % (split, i) for i in range(n_shards)]
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
class WebCIFAR20DatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-20 WebDataset for image classification.

    Parameters:
        train_shards (int): Number of training split shards.
        val_shards (int): Number of validation split shards.
    """
    train_shards: int = hp.optional('Training split shards', default=16)
    val_shards: int = hp.optional('Validation split shards', default=8)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        cifar20_mean = 0.5071, 0.4867, 0.4408
        cifar20_std = 0.2675, 0.2565, 0.2761

        if self.is_train:
            split = 'train'
            size = 50_000
            n_shards = self.train_shards
        else:
            split = 'val'
            size = 10_000
            n_shards = self.val_shards
        size = size - size % n_shards

        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=size,
                data_shape=[3, 32, 32],
                num_classes=20,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
        else:
            if self.is_train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar20_mean, cifar20_std),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cifar20_mean, cifar20_std),
                ])

            urls = ['/datasets/web_cifar20/%s_%05d.tar' % (split, i) for i in range(n_shards)]
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
class WebCIFAR100DatasetHparams(WebDatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the CIFAR-100 WebDataset for image classification.

    Parameters:
        train_shards (int): Number of training split shards.
        val_shards (int): Number of validation split shards.
    """
    train_shards: int = hp.optional('Training split shards', default=16)
    val_shards: int = hp.optional('Validation split shards', default=8)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        cifar100_mean = 0.5071, 0.4867, 0.4408
        cifar100_std = 0.2675, 0.2565, 0.2761

        if self.is_train:
            split = 'train'
            size = 50_000
            n_shards = self.train_shards
        else:
            split = 'val'
            size = 10_000
            n_shards = self.val_shards
        size = size - size % n_shards

        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=size,
                data_shape=[3, 32, 32],
                num_classes=100,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )
        else:
            if self.is_train:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(cifar100_mean, cifar100_std),
                ])

            urls = ['/datasets/web_cifar100/%s_%05d.tar' % (split, i) for i in range(n_shards)]
            size_per_device = size // dist.get_world_size()
            dataset = WebDataset(urls, cache_dir=self.dataset_cache_dir,
                                 cache_verbose=self.dataset_cache_verbose)
            dataset = dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
            dataset = dataset.with_epoch(size_per_device).with_length(size_per_device)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)
