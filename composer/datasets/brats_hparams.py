# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""BraTS (Brain Tumor Segmentation) dataset hyperparameters."""

from dataclasses import dataclass
from typing import Optional

import torch
import yahp as hp

from composer.datasets.brats import PytTrain, PytVal, get_data_split
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.utils import dist


def _my_collate(batch):
    """Custom collate function to handle images with different depths."""
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]

    return [torch.Tensor(data), torch.Tensor(target)]


@dataclass
class BratsDatasetHparams(DatasetHparams):
    """Defines an instance of the BraTS dataset for image segmentation.

    Args:
        oversampling (float): The oversampling ratio to use. Default: ``0.33``.
    """

    oversampling: float = hp.optional('oversampling', default=0.33)
    is_train: bool = hp.optional('Whether to load the training data (the default) or validation data.', default=True)
    datadir: Optional[str] = hp.optional('The path to the data directory', default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):

        oversampling = self.oversampling

        if self.datadir is None:
            raise ValueError('datadir must be specified.')
        x_train, y_train, x_val, y_val = get_data_split(self.datadir)
        dataset = PytTrain(x_train, y_train, oversampling) if self.is_train else PytVal(x_val, y_val)
        collate_fn = None if self.is_train else _my_collate
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=collate_fn,
        )
