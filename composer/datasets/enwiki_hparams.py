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
            Default: ``'s3://mosaicml-internal-dataset-enwiki-20200101/mds/2/'``
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-enwiki/'``
        split (str): What split of the dataset to use. Either ``'train'`` or ``'val'``. Default: ``'train'``.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text with. Default:
            ``'bert-base-uncased'``.
        max_seq_len (int): The max sequence length of each token sample. Default: ``512``.
        group_method (str): How to group text samples into token samples. Currently only `truncate` is supported.
        mlm (bool): Whether or not to use masked language modeling. Default: ``False``.
        mlm_probability (float): If ``mlm==True``, the probability that tokens are masked. Default: ``0.15``.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
    """

    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-enwiki-20200101/mds/2/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-enwiki/')
    split: str = hp.optional('What split of the dataset to use. Either `train` or `val`.', default='train')
    tokenizer_name: str = hp.optional('The name of the HuggingFace tokenizer to preprocess text with.',
                                      default='bert-base-uncased')
    max_seq_len: int = hp.optional('The max sequence length of each token sample.', default=512)
    group_method: str = hp.optional(
        'How to group text samples into token samples. Currently only `truncate` is supported.', default='truncate')
    mlm: bool = hp.optional('Whether or not to use masked language modeling.', default=False)
    mlm_probability: float = hp.optional('If `mlm==True`, the probability that tokens are masked.', default=0.15)
    max_retries: int = hp.optional('Number of download re-attempts before giving up.', default=2)
    timeout: float = hp.optional('How long to wait for shard to download before raising an exception.', default=120)

    def validate(self):
        if self.split not in ['train', 'val']:
            raise ValueError(f"Unknown split: '{self.split}'")
        if self.tokenizer_name is None:
            raise ValueError(f"Must provide 'tokenizer_name'")
        if self.max_seq_len is None or self.max_seq_len <= 0:
            raise ValueError(f"Must provide 'max_seq_len' > 0")
        if self.group_method not in ['truncate']:
            raise ValueError(f"Unknown group_method: '{self.group_method}'. Currently only 'truncate' is supported.")
        if self.mlm and self.mlm_probability <= 0:
            raise ValueError("Must provide a positive 'mlm_probability' when using masked language modeling.")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

        # Get StreamingEnWiki dataset
        try:
            from streaming.text import EnWiki
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e
        dataset = EnWiki(tokenizer_name=self.tokenizer_name,
                         max_seq_len=self.max_seq_len,
                         group_method=self.group_method,
                         local=self.local,
                         remote=self.remote,
                         split=self.split,
                         shuffle=self.shuffle,
                         retry=self.max_retries,
                         timeout=self.timeout,
                         batch_size=batch_size)

        # Get collate_fn
        collate_fn = transformers.DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer,
                                                                  mlm=self.mlm,
                                                                  mlm_probability=self.mlm_probability)

        return DataSpec(
            dataloader=dataloader_hparams.initialize_object(
                dataset=dataset,  # type: ignore
                batch_size=batch_size,
                sampler=None,
                drop_last=self.drop_last,
                collate_fn=collate_fn),
            device_transforms=None)
