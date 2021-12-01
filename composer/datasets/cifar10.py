# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import Optional

import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDatasetHparams
from composer.utils import ddp


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

    is_train: Optional[bool] = hp.optional(
        "whether to load the training or validation dataset. Required if synthetic is not None.", default=None)
    synthetic: Optional[SyntheticBatchPairDatasetHparams] = hp.optional(
        "If specified, synthetic data will be generated. The datadir argument is ignored", default=None)
    num_total_batches: Optional[int] = hp.optional("num total batches", default=None)
    datadir: Optional[str] = hp.optional("data directory. Required if synthetic is not None.", default=None)
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch", default=True)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch", default=True)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        cifar10_mean, cifar10_std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

        if self.synthetic is not None:
            if self.num_total_batches is None:
                raise ValueError("num_total_batches is required if synthetic is True")
            dataset = self.synthetic.initialize_object(
                total_dataset_size=self.num_total_batches * batch_size,
                data_shape=[3, 32, 32],
                num_classes=10,
            )
            if self.shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)

        else:
            if self.datadir is None:
                raise ValueError("datadir is required if synthetic is None")

            if self.is_train is True:
                transformation = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ])
            elif self.is_train is False:
                transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
                ])
            else:
                raise ValueError("is_train is required if synthetic is None")
            dataset = CIFAR10(
                self.datadir,
                train=self.is_train,
                download=self.download,
                transform=transformation,
            )
            if self.num_total_batches is not None:
                size = batch_size * self.num_total_batches * ddp.get_world_size()
                dataset = torch.utils.data.Subset(dataset, list(range(size)))
            sampler = ddp.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)
