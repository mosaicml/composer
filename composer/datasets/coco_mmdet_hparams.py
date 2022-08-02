# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import yahp as hp

from composer.datasets.coco_mmdet import coco_mmdet, mmdet_collate
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams, DataSpec
from composer.utils import dist


@dataclass
class COCOMMDetHparams(DatasetHparams):
    """DatasetHparams for creating a coco dataset in mmdetection format.

    Args:
        path (str): path to unzipped coco data. Default: ``''/data/coco'```.
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
    """
    path: str = hp.optional('path to coco dataset', default='/data/coco')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        dataset = coco_mmdet(self.path, self.split)
        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(dataset,
                                                    batch_size=batch_size,
                                                    sampler=sampler,
                                                    drop_last=self.drop_last,
                                                    collate_fn=mmdet_collate)
