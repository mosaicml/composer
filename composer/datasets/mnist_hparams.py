# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""MNIST image classification dataset hyperparameters.

The MNIST dataset is a collection of labeled 28x28 black and white images of handwritten examples of the numbers 0-9.
See the `wikipedia entry <https://en.wikipedia.org/wiki/MNIST_database>`_ for more details.
"""

from dataclasses import dataclass
from typing import Optional

import yahp as hp
from torchvision import datasets, transforms

from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.utils import dist

__all__ = ['MNISTDatasetHparams']


@dataclass
class MNISTDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
    """Defines an instance of the MNIST dataset for image classification.

    Args:
        download (bool, optional): Whether to download the dataset, if needed. Default:
            ``True``.
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
    """
    download: bool = hp.optional('whether to download the dataset, if needed', default=True)
    is_train: bool = hp.optional('Whether to load the training data (the default) or validation data.', default=True)
    datadir: Optional[str] = hp.optional('The path to the data directory', default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        if self.use_synthetic:
            dataset = SyntheticBatchPairDataset(
                total_dataset_size=60_000 if self.is_train else 10_000,
                data_shape=[1, 28, 28],
                num_classes=10,
                num_unique_samples_to_create=self.synthetic_num_unique_samples,
                device=self.synthetic_device,
                memory_format=self.synthetic_memory_format,
            )

        else:
            if self.datadir is None:
                raise ValueError('datadir is required if synthetic is False')

            transform = transforms.Compose([transforms.ToTensor()])

            with dist.run_local_rank_zero_first():
                dataset = datasets.MNIST(
                    self.datadir,
                    train=self.is_train,
                    download=dist.get_local_rank() == 0 and self.download,
                    transform=transform,
                )

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return dataloader_hparams.initialize_object(dataset=dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last)
