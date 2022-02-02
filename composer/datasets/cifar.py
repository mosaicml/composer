# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10
from webdataset import WebDataset

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, JpgClsWebDatasetHparams, SyntheticHparamsMixin
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
class CIFAR10WebDatasetHparams(JpgClsWebDatasetHparams):
    """Defines an instance of the CIFAR-10 WebDataset for image classification."""

    dataset_name = 'cifar10'
    n_train_samples = 50_000
    n_val_samples = 10_000
    height = 32
    width = 32
    n_classes = 10
    channel_means = 0.4914, 0.4822, 0.4465
    channel_stds = 0.247, 0.243, 0.261


@dataclass
class CIFAR20WebDatasetHparams(JpgClsWebDatasetHparams):
    """Defines an instance of the CIFAR-20 WebDataset for image classification."""

    dataset_name = 'cifar20'
    n_train_samples = 50_000
    n_val_samples = 10_000
    height = 32
    width = 32
    n_classes = 20
    channel_means = 0.5071, 0.4867, 0.4408
    channel_stds = 0.2675, 0.2565, 0.2761


@dataclass
class CIFAR100WebDatasetHparams(JpgClsWebDatasetHparams):
    """Defines an instance of the CIFAR-100 WebDataset for image classification."""

    dataset_name = 'cifar100'
    n_train_samples = 50_000
    n_val_samples = 10_000
    height = 32
    width = 32
    n_classes = 100
    channel_means = 0.5071, 0.4867, 0.4408
    channel_stds = 0.2675, 0.2565, 0.2761
