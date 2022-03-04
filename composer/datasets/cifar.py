# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import List

import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin, WebDatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.webdataset import load_webdataset
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
class CIFARWebDatasetHparams(WebDatasetHparams):
    """Common functionality for CIFAR WebDatasets.

    Parameters:
        remote (str): S3 bucket or root directory where dataset is stored.
        name (str): Key used to determine where dataset is cached on local filesystem.
        n_train_samples (int): Number of training samples.
        n_val_samples (int): Number of validation samples.
        height (int): Sample image height in pixels.
        width (int): Sample image width in pixels.
        n_classes (int): Number of output classes.
        channel_means (list of float): Channel means for normalization.
        channel_stds (list of float): Channel stds for normalization.
    """

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
        preprocess = lambda dataset: dataset.decode('pil').map_dict(jpg=transform).to_tuple('jpg', 'cls')
        dataset = load_webdataset(self.remote, self.name, split, self.webdataset_cache_dir, self.webdataset_cache_verbose,
                                  self.shuffle, self.shuffle_buffer, preprocess, dist.get_world_size(),
                                  dataloader_hparams.num_workers, batch_size, self.drop_last)
        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    drop_last=self.drop_last)


@dataclass
class CIFAR10WebDatasetHparams(CIFARWebDatasetHparams):
    """Defines an instance of the CIFAR-10 WebDataset for image classification."""

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-cifar10')
    name: str = hp.optional('WebDataset local cache name', default='cifar10')
    n_train_samples: int = 50_000
    n_val_samples: int = 10_000
    height: int = 32
    width: int = 32
    n_classes: int = 10
    channel_means: List[float] = 0.4914, 0.4822, 0.4465
    channel_stds: List[float] = 0.247, 0.243, 0.261


@dataclass
class CIFAR20WebDatasetHparams(CIFARWebDatasetHparams):
    """Defines an instance of the CIFAR-20 WebDataset for image classification."""

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-cifar20')
    name: str = hp.optional('WebDataset local cache name', default='cifar20')
    n_train_samples: int = 50_000
    n_val_samples: int = 10_000
    height: int = 32
    width: int = 32
    n_classes: int = 20
    channel_means: List[float] = 0.5071, 0.4867, 0.4408
    channel_stds: List[float] = 0.2675, 0.2565, 0.2761


@dataclass
class CIFAR100WebDatasetHparams(CIFARWebDatasetHparams):
    """Defines an instance of the CIFAR-100 WebDataset for image classification."""

    remote: str = hp.optional('WebDataset S3 bucket name', default='s3://mosaicml-internal-dataset-cifar100')
    name: str = hp.optional('WebDataset local cache name', default='cifar100')
    n_train_samples: int = 50_000
    n_val_samples: int = 10_000
    height: int = 32
    width: int = 32
    n_classes: int = 100
    channel_means: List[float] = 0.5071, 0.4867, 0.4408
    channel_stds: List[float] = 0.2675, 0.2565, 0.2761
