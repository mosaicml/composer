# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset hyperparameters."""
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import yahp as hp

from composer.core.data_spec import DataSpec
from composer.datasets.c4 import build_c4_dataloader, build_streamingc4_dataloader
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams

log = logging.getLogger(__name__)

__all__ = ['C4DatasetHparams', 'StreamingC4Hparams']


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
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
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

        # Get StreamingC4 dataset
        return build_streamingc4_dataloader(
            batch_size=batch_size,
            remote=self.remote,
            local=self.local,
            split=self.split,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            tokenizer_name=self.tokenizer_name,
            max_seq_len=self.max_seq_len,
            group_method=self.group_method,
            mlm=self.mlm,
            mlm_probability=self.mlm_probability,
            max_retries=self.max_retries,
            timeout=self.timeout,
            version=self.version,
            **asdict(dataloader_hparams),
        )


@dataclass
class C4DatasetHparams(DatasetHparams):
    """Builds a :class:`.DataSpec` for the C4 (Colossal Cleaned Common Crawl) dataset.

    Args:
        split (str): What split of the dataset to use. Either ``'train'`` or ``'validation'``. Default: ``'train'``.
        num_samples (int): The number of post-processed token samples, used to set epoch size of the
            :class:`torch.utils.data.IterableDataset`. Default: ``None``.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text with. Default: ``'bert-base-uncased'``.
        max_seq_len (int): The max sequence length of each token sample. Default: ``512``.
        group_method (str): How to group text samples into token samples. Either `truncate` or `concat`.
            Default: ``'truncate'``.
        mlm (bool): Whether or not to use masked language modeling. Default: ``False``.
        mlm_probability (float): If ``mlm==True``, the probability that tokens are masked. Default: ``0.15``.
        shuffle (bool): Whether to shuffle the samples in the dataset. Currently, shards are assigned and consumed with
            deterministic per-device shard order, but shuffling affects the order of samples via (per-device) shuffle
            buffers. Default: ``False``.
        shuffle_buffer_size (int): If ``shuffle=True``, samples are read into a buffer of this size (per-device), and
            randomly sampled from there to produce shuffled samples. Default: ``10000``.
        seed (int): If ``shuffle=True``, what seed to use for shuffling operations. Default: ``5``.
        drop_last (bool): Whether to drop the last samples for the last batch. Default: ``True``.
    Returns:
        DataLoader: A PyTorch :class:`~torch.utils.data.DataLoader` object.
    """

    split: str = hp.optional('What split of the dataset to use. Either `train` or `validation`.', default='train')
    num_samples: Optional[int] = hp.optional(
        'The number of post-processed token samples, used to set epoch size of the IterableDataset.', default=None)
    tokenizer_name: str = hp.optional('The name of the HuggingFace tokenizer to preprocess text with.',
                                      default='bert-base-uncased')
    max_seq_len: int = hp.optional('The max sequence length of each token sample.', default=512)
    group_method: str = hp.optional('How to group text samples into token samples. Either `truncate` or `concat`.',
                                    default='truncate')
    mlm: bool = hp.optional('Whether or not to use masked language modeling.', default=False)
    mlm_probability: float = hp.optional('If `mlm==True`, the probability that tokens are masked.', default=0.15)
    shuffle: bool = hp.optional(
        'Whether to shuffle the samples in the dataset. Currently, shards are assigned and consumed with deterministic per-device shard order, but shuffling affects the order of samples via (per-device) shuffle buffers.',
        default=True)
    shuffle_buffer_size: int = hp.optional(
        'If `shuffle=True`, samples are read into a buffer of this size (per-device), and randomly sampled from there to produce shuffled samples.',
        default=10000)
    seed: int = hp.optional('If `shuffle=True`, what seed to use for shuffling operations.', default=5)
    drop_last: bool = hp.optional('Whether to drop the last samples for the last batch.', default=True)

    def validate(self):
        if self.split not in ['train', 'validation']:
            raise ValueError(f"Unknown split: '{self.split}'")
        if self.num_samples is None or self.num_samples <= 0:
            raise ValueError(f"Must provide 'num_samples' > 0")
        if self.tokenizer_name is None:
            raise ValueError(f"Must provide 'tokenizer_name'")
        if self.max_seq_len is None or self.max_seq_len <= 0:
            raise ValueError(f"Must provide 'max_seq_len' > 0")
        if self.group_method not in ['truncate', 'concat']:
            raise ValueError(f"Unknown group_method: '{self.group_method}'. Must be 'truncate' or 'concat'")
        if self.mlm and self.mlm_probability <= 0:
            raise ValueError("Must provide a positive 'mlm_probability' when using masked language modeling.")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:

        return build_c4_dataloader(
            num_samples=self.num_samples,  # type: ignore
            batch_size=batch_size,
            split=self.split,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            tokenizer_name=self.tokenizer_name,
            max_seq_len=self.max_seq_len,
            group_method=self.group_method,
            mlm=self.mlm,
            mlm_probability=self.mlm_probability,
            seed=self.seed,
            shuffle_buffer_size=self.shuffle_buffer_size,
            **asdict(dataloader_hparams),
        )
