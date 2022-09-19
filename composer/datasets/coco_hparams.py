# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""COCO (Common Objects in Context) dataset hyperparameters."""
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.core import DataSpec
from composer.datasets.coco import StreamingCOCO, build_coco_detection_dataloader, split_coco_batch
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.models.ssd.utils import SSDTransformer, dboxes300_coco
from composer.utils import warn_streaming_dataset_deprecation
from composer.utils.import_helpers import MissingConditionalImportError

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
        version (int): Which version of streaming to use. Default: ``1``.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-coco/mds/1/```
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-coco/```
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
    """

    version: int = hp.optional('Version of streaming (1 or 2)', default=1)
    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-coco/mds/1/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-coco/')
    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val']", default='train')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        if self.version == 1:
            warn_streaming_dataset_deprecation(old_version=self.version, new_version=2)
            dataset = StreamingCOCO(remote=self.remote,
                                    local=self.local,
                                    split=self.split,
                                    shuffle=self.shuffle,
                                    batch_size=batch_size)
        elif self.version == 2:
            try:
                from streaming.vision import COCO
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='streaming',
                                                    conda_package='mosaicml-streaming') from e
            # Define custom transforms
            dboxes = dboxes300_coco()
            input_size = 300
            if self.split == 'train':
                transform = SSDTransformer(dboxes, (input_size, input_size), val=False, num_cropping_iterations=1)
            else:
                transform = SSDTransformer(dboxes, (input_size, input_size), val=True)
            dataset = COCO(local=self.local,
                           remote=self.remote,
                           split=self.split,
                           shuffle=self.shuffle,
                           transform=transform,
                           batch_size=batch_size)
        else:
            raise ValueError(f'Invalid streaming version: {self.version}')
        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            drop_last=self.drop_last,
            batch_size=batch_size,
            sampler=None,
            collate_fn=None,
        ),
                        split_batch=split_coco_batch)
