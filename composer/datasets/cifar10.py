# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass

import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.datasets.hparams import DataloaderSpec, DatasetHparams


@dataclass
class CIFAR10DatasetHparams(DatasetHparams):
    """Defines an instance of the CIFAR-10 dataset for image classification.
    
    Parameters:
        is_train (bool): Whether to load the training or validation dataset.
        datadir (str): Data directory to use.
        download (bool): Whether to download the dataset, if needed.
        drop_last (bool): Whether to drop the last samples for the last batch.
        shuffle (bool): Whether to shuffle the dataset for each epoch.
    """

    is_train: bool = hp.required("whether to load the training or validation dataset")
    datadir: str = hp.required("data directory")
    download: bool = hp.required("whether to download the dataset, if needed")
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def initialize_object(self) -> DataloaderSpec:
        cifar10_mean, cifar10_std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
        datadir = self.datadir

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

        return DataloaderSpec(
            dataset=CIFAR10(
                datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            ),
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )
