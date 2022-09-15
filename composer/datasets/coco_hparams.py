# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset hyperparameters."""
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.core import DataSpec
from composer.datasets.coco import StreamingCOCO, build_coco_detection_dataloader, split_coco_batch
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams

__all__ = ['COCODatasetHparams', 'StreamingCOCOHparams']


@dataclass
class COCODatasetHparams(DatasetHparams):
    """Defines an instance of the COCO Dataset.

    Args:
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
    """

    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val'].", default='train')
    datadir: Optional[str] = hp.optional('The path to the data directory.', default=None)
    input_size: int = hp.optional('Input image size, keep at 300 if using with SSD300.', default=300)

    def validate(self):
        if self.datadir is None:
            raise ValueError('datadir must specify the path to the COCO Detection dataset.')

        if self.split not in ['train', 'val']:
            raise ValueError(f"split value {self.split} must be one of ['train', 'val'].")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):

        self.validate()

        return build_coco_detection_dataloader(
            batch_size=batch_size,
            datadir=self.datadir,  #type: ignore
            split=self.split,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            input_size=self.input_size,
            **asdict(dataloader_hparams))


@dataclass
class StreamingCOCOHparams(DatasetHparams):
    """DatasetHparams for creating an instance of StreamingCOCO.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-coco/mds/1/```
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-coco/```
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-coco/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-coco/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        dataset = StreamingCOCO(remote=self.remote,
                                local=self.local,
                                split=self.split,
                                shuffle=self.shuffle,
                                batch_size=batch_size)
        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            drop_last=self.drop_last,
            batch_size=batch_size,
            sampler=None,
            collate_fn=None,
        ),
                        split_batch=split_coco_batch)
