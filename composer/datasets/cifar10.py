# Copyright 2021 MosaicML. All Rights Reserved.

import textwrap
from dataclasses import dataclass
from typing import Optional

import torch.utils.data
import yahp as hp
from torchvision import transforms
from torchvision.datasets import CIFAR10

from composer.core.types import DataLoader
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import (DatadirHparamsMixin, DatasetHparams, DropLastHparamsMixin, IsTrainHparamsMixin,
                                       NumTotalBatchesHparamsMixin, ShuffleHparamsMixin, SyntheticBatchesHparamsMixin)
from composer.datasets.synthetic import SyntheticBatchPairDatasetHparams
from composer.utils import ddp


@dataclass
class CIFAR10DatasetHparams(DatasetHparams, ShuffleHparamsMixin, DropLastHparamsMixin, DatadirHparamsMixin,
                            NumTotalBatchesHparamsMixin, SyntheticBatchesHparamsMixin, IsTrainHparamsMixin):
    """Defines an instance of the CIFAR-10 dataset for image classification.
    
    Parameters:
        download (bool): Whether to download the dataset, if needed.
    """
    download: bool = hp.optional("whether to download the dataset, if needed", default=True)
    synthetic: Optional[SyntheticBatchPairDatasetHparams] = hp.optional(
        textwrap.dedent("""Parameters to use for synthetic data generation.
            If None (the default), then real data will be used."""),
        default=None)

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
            if self.num_total_batches is not None:
                size = batch_size * self.num_total_batches * ddp.get_world_size()
                dataset = torch.utils.data.Subset(dataset, list(range(size)))
            sampler = ddp.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)
