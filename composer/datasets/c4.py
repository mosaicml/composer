# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset.

This dataset is a colossal, cleaned version of Common Crawl's web crawl corpus and it is based on the `Common Crawl
<https://commoncrawl.org>`_ dataset.
"""
import logging
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from composer.core import DataSpec
from composer.utils import MissingConditionalImportError, dist

log = logging.getLogger(__name__)

__all__ = ['build_streaming_c4_dataloader']


def build_streaming_c4_dataloader(
    global_batch_size: int,
    remote: str = 's3://mosaicml-internal-dataset-c4/mds/2/',
    local: str = '/tmp/mds-cache/mds-c4/',
    split: str = 'train',
    shuffle: bool = True,
    drop_last: bool = True,
    tokenizer_name: str = 'bert-base-uncased',
    max_seq_len: int = 512,
    group_method: str = 'truncate',
    mlm: bool = False,
    mlm_probability: float = 0.15,
    predownload: Optional[int] = 100_000,
    keep_zip: Optional[bool] = None,
    download_retry: int = 2,
    download_timeout: float = 60,
    validate_hash: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs: Dict[str, Any],
):
    """Builds a :class:`.DataSpec` for the StreamingC4 (Colossal Cleaned Common Crawl) dataset.

    Args:
        global_batch_size (int): Global batch size.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
            Default: ``'s3://mosaicml-internal-dataset-c4/mds/2/'``
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-c4/'``
        split (str): What split of the dataset to use. Either ``'train'`` or ``'val'``.
            Default: ``'train'``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text with. Default:
            ``'bert-base-uncased'``.
        max_seq_len (int): The max sequence length of each token sample. Default: ``512``.
        group_method (str): How to group text samples into token samples. Currently only `truncate` is supported.
        mlm (bool): Whether or not to use masked language modeling. Default: ``False``.
        mlm_probability (float): If ``mlm==True``, the probability that tokens are masked. Default: ``0.15``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """

    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()

    try:
        from streaming.text import StreamingC4
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e

    dataset = StreamingC4(
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        group_method=group_method,
        local=local,
        remote=remote,
        split=split,
        shuffle=shuffle,
        predownload=predownload,
        keep_zip=keep_zip,
        download_retry=download_retry,
        download_timeout=download_timeout,
        validate_hash=validate_hash,
        shuffle_seed=shuffle_seed,
        num_canonical_nodes=num_canonical_nodes,
        batch_size=batch_size,
    )

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=mlm,
        mlm_probability=mlm_probability,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )

    return DataSpec(dataloader=dataloader)
