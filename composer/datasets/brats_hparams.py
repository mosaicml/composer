# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""BraTS (Brain Tumor Segmentation) dataset hyperparameters."""

from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.datasets.brats import build_brats_dataloader
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams


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
        if self.datadir is None:
            raise ValueError('datadir must be specified.')

        return build_brats_dataloader(datadir=self.datadir,
                                      batch_size=batch_size,
                                      oversampling=self.oversampling,
                                      is_train=self.is_train,
                                      drop_last=self.drop_last,
                                      shuffle=self.shuffle,
                                      **asdict(dataloader_hparams))
