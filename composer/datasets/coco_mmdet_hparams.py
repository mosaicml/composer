# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import yahp as hp

from composer.datasets.coco_mmdet import coco_mmdet, mmdet_collate
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams, DataSpec


@dataclass
class COCOMMDetHparams(DatasetHparams):

    path: str = hp.optional('path to coco dataset')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')


    def initialize_object(self, batch_size:int, dataloader_hparams:DataLoaderHparams) -> DataSpec:
        dataset = coco_mmdet(self.path, self.split)
        return dataloader_hparams.initialize_object(dataset,
                                                batch_size=batch_size,
                                                sampler=None,
                                                drop_last=self.drop_last
                                                collate_fn=mmdet_collate)
