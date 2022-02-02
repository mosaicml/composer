# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
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
        cifar10_mean, cifar10_std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

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
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ])
            else:
                transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
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
