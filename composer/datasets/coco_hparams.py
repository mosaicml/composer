# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset hyperparameters."""
import os
from dataclasses import dataclass
from typing import Optional

import yahp as hp

from composer.core import DataSpec
from composer.datasets.coco import COCODetection, StreamingCOCO, split_dict_fn
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.models.ssd.utils import SSDTransformer, dboxes300_coco
from composer.utils import dist

__all__ = ['COCODatasetHparams', 'StreamingCOCOHparams']


@dataclass
class COCODatasetHparams(DatasetHparams):
    """Defines an instance of the COCO Dataset.

    Args:
        datadir (str): The path to the data directory.
        is_train (bool): Whether to load the training data or validation data. Default:
            ``True``.
    """

    is_train: bool = hp.optional('Whether to load the training data (the default) or validation data.', default=True)
    datadir: Optional[str] = hp.optional('The path to the data directory', default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):

        if self.datadir is None:
            raise ValueError('datadir is required.')

        dboxes = dboxes300_coco()

        input_size = 300
        train_trans = SSDTransformer(dboxes, (input_size, input_size), val=False, num_cropping_iterations=1)
        val_trans = SSDTransformer(dboxes, (input_size, input_size), val=True)
        data = self.datadir

        val_annotate = os.path.join(data, 'annotations/instances_val2017.json')
        val_coco_root = os.path.join(data, 'val2017')

        train_annotate = os.path.join(data, 'annotations/instances_train2017.json')
        train_coco_root = os.path.join(data, 'train2017')

        train_coco = COCODetection(train_coco_root, train_annotate, train_trans)
        val_coco = COCODetection(val_coco_root, val_annotate, val_trans)

        if self.is_train:
            return DataSpec(dataloader=dataloader_hparams.initialize_object(
                dataset=train_coco,
                batch_size=batch_size,
                sampler=dist.get_sampler(train_coco, drop_last=self.drop_last, shuffle=self.shuffle),
                drop_last=self.drop_last,
            ),
                            split_batch=split_dict_fn)
        else:
            return DataSpec(dataloader=dataloader_hparams.initialize_object(
                dataset=val_coco,
                drop_last=self.drop_last,
                batch_size=batch_size,
                sampler=None,
            ),
                            split_batch=split_dict_fn)


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
                        split_batch=split_dict_fn)
