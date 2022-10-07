# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""English Wikipedia 2020-01-01 dataset hyperparameters."""

import logging
from dataclasses import dataclass

import yahp as hp

from composer.core.data_spec import DataSpec
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['StreamingEnWikiHparams']


@dataclass
class StreamingEnWikiHparams(DatasetHparams):
    """Builds a :class:`.DataSpec` for the StreamingEnWiki (English Wikipedia 2020-01-01) dataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-enwiki-20200101/mds/2b/'``
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-enwiki/'``
        split (str): What split of the dataset to use. Either ``'train'`` or ``'val'``. Default: ``'train'``.
        mlm (bool): Whether or not to use masked language modeling. Default: ``False``.
        mlm_probability (float): If ``mlm==True``, the probability that tokens are masked. Default: ``0.15``.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-enwiki-20200101/mds/2b/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-enwiki/')
    split: str = hp.optional('What split of the dataset to use. Either `train` or `val`.', default='train')
    mlm: bool = hp.optional('Whether or not to use masked language modeling.', default=False)
    mlm_probability: float = hp.optional('If `mlm==True`, the probability that tokens are masked.', default=0.15)
    max_retries: int = hp.optional('Number of download re-attempts before giving up.', default=2)
    timeout: float = hp.optional('How long to wait for shard to download before raising an exception.', default=120)

    def validate(self):
        if self.split not in ['train', 'val']:
            raise ValueError(f"Unknown split: '{self.split}'")
        if self.mlm and self.mlm_probability <= 0:
            raise ValueError("Must provide a positive 'mlm_probability' when using masked language modeling.")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        # Get StreamingEnWiki dataset
        try:
            from streaming.text import EnWiki
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e
        dataset = EnWiki(local=self.local,
                         remote=self.remote,
                         split=self.split,
                         shuffle=self.shuffle,
                         retry=self.max_retries,
                         timeout=self.timeout,
                         batch_size=batch_size)

        return DataSpec(
            dataloader=dataloader_hparams.initialize_object(
                dataset=dataset,  # type: ignore
                batch_size=batch_size,
                sampler=None,
                drop_last=self.drop_last,
                collate_fn=None),
            device_transforms=None)
