# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset hyperparameters."""
import logging
from dataclasses import dataclass

import yahp as hp

from composer.core.data_spec import DataSpec
from composer.datasets.c4 import StreamingC4
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.utils import warn_streaming_dataset_deprecation
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['StreamingC4Hparams']


@dataclass
class StreamingC4Hparams(DatasetHparams):
    """Builds a :class:`.DataSpec` for the StreamingC4 (Colossal Cleaned Common Crawl) dataset.

    Args:
        version (int): Which version of streaming to use. Default: ``2``.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-c4/mds/2/'``
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-c4/'``
        split (str): What split of the dataset to use. Either ``'train'`` or ``'val'``. Default: ``'train'``.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text with. Default:
            ``'bert-base-uncased'``.
        max_seq_len (int): The max sequence length of each token sample. Default: ``512``.
        group_method (str): How to group text samples into token samples. Currently only `truncate` is supported.
        mlm (bool): Whether or not to use masked language modeling. Default: ``False``.
        mlm_probability (float): If ``mlm==True``, the probability that tokens are masked. Default: ``0.15``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Default: ``False``.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
        drop_last (bool): Whether to drop the last samples for the last batch. Default: ``True``.
    """

    version: int = hp.optional('Version of streaming (1 or 2)', default=2)
    remote: str = hp.optional('Remote directory (S3 or local filesystem) where dataset is stored',
                              default='s3://mosaicml-internal-dataset-c4/mds/2/')
    local: str = hp.optional('Local filesystem directory where dataset is cached during operation',
                             default='/tmp/mds-cache/mds-c4/')
    split: str = hp.optional('What split of the dataset to use. Either `train` or `val`.', default='train')
    tokenizer_name: str = hp.optional('The name of the HuggingFace tokenizer to preprocess text with.',
                                      default='bert-base-uncased')
    max_seq_len: int = hp.optional('The max sequence length of each token sample.', default=512)
    group_method: str = hp.optional(
        'How to group text samples into token samples. Currently only `truncate` is supported.', default='truncate')
    mlm: bool = hp.optional('Whether or not to use masked language modeling.', default=False)
    mlm_probability: float = hp.optional('If `mlm==True`, the probability that tokens are masked.', default=0.15)
    shuffle: bool = hp.optional('Whether to iterate over the samples in randomized order.', default=False)
    max_retries: int = hp.optional('Number of download re-attempts before giving up.', default=2)
    timeout: float = hp.optional('How long to wait for shard to download before raising an exception.', default=120)
    drop_last: bool = hp.optional('Whether to drop the last samples for the last batch.', default=True)

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

        # Get StreamingC4 dataset
        if self.version == 1:
            warn_streaming_dataset_deprecation(old_version=self.version, new_version=2)
            dataset = StreamingC4(remote=self.remote,
                                  local=self.local,
                                  split=self.split,
                                  shuffle=self.shuffle,
                                  tokenizer_name=self.tokenizer_name,
                                  max_seq_len=self.max_seq_len,
                                  group_method=self.group_method,
                                  max_retries=self.max_retries,
                                  timeout=self.timeout,
                                  batch_size=batch_size)
        elif self.version == 2:
            try:
                from streaming.text import C4
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='streaming',
                                                    conda_package='mosaicml-streaming') from e
            dataset = C4(tokenizer_name=self.tokenizer_name,
                         max_seq_len=self.max_seq_len,
                         group_method=self.group_method,
                         local=self.local,
                         remote=self.remote,
                         split=self.split,
                         shuffle=self.shuffle,
                         retry=self.max_retries,
                         timeout=self.timeout,
                         batch_size=batch_size)
        else:
            raise ValueError(f'Invalid streaming version: {self.version}')

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
