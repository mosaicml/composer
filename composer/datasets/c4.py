# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset.

This dataset is a colossal, cleaned version of Common Crawl's web crawl corpus and it is based on the `Common Crawl
<https://commoncrawl.org>`_ dataset.
"""
import logging
import os
from typing import Any, Dict, Iterator, Optional

from torch.utils.data import DataLoader

from composer.core.data_spec import DataSpec
from composer.datasets.streaming import StreamingDataset
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['StreamingC4']


class StreamingC4(StreamingDataset):
    """
    Implementation of the C4 (Colossal Cleaned Common Crawl) dataset using StreamingDataset V1.
    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Supports 'truncate' or 'concat'.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 max_retries: int = 2,
                 timeout: float = 120,
                 batch_size: Optional[int] = None):

        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate', 'concat']:
            raise ValueError(f"group_method='{group_method}' must be one of ['truncate', 'concat'].")

        # Build StreamingDataset
        decoders = {
            'text': self._decode,
            'timestamp': self._decode,
            'url': self._decode,
        }
        super().__init__(remote=os.path.join(remote, split),
                         local=os.path.join(local, split),
                         shuffle=shuffle,
                         decoders=decoders,
                         max_retries=max_retries,
                         timeout=timeout,
                         batch_size=batch_size)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    # How to decode binary data from .mds files to python strings
    def _decode(self, data: bytes) -> str:
        return data.decode('utf-8')

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        elif self.group_method == 'concat':
            truncation = False
            padding = False
            max_length = None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")
        return self.tokenizer(text_sample['text'], truncation=truncation, padding=padding, max_length=max_length)

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        return token_sample

    # Define iterable over samples
    # Usually this can be left alone and inherited directly from super() class StreamingDataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the token sample.
    # If group_method=='concat', then we keep fetching token samples until we fill up max_seq_len.
    def __iter__(self) -> Iterator[Any]:
        if self.group_method == 'truncate':
            iterator = super().__iter__()
            yield from iterator

        elif self.group_method == 'concat':
            buffer = {}
            while True:
                iterator = super().__iter__()
                for sample in iterator:

                    for k, v in sample.items():
                        buffer[k] = buffer.get(k, []) + v
                    while len(buffer['input_ids']) >= self.max_seq_len:
                        concat_sample = {}
                        for k, v in buffer.items():
                            concat_sample[k] = v[:self.max_seq_len]
                            buffer[k] = v[self.max_seq_len:]
                        yield concat_sample
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")

    # Define length
    # Usually this can be left alone and inherited directly from super() class StreamingDataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the # samples.
    # If group_method=='concat', we repeat forever, and we don't have a defined length.
    def __len__(self) -> Optional[int]:
        if self.group_method == 'truncate':
            return super().__len__()
        elif self.group_method == 'concat':
            return None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")


def build_streamingc4_dataloader(
    batch_size: int,
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
    max_retries: int = 2,
    timeout: float = 120,
    version: int = 2,
    **dataloader_kwargs,
):
    """Builds a :class:`.DataSpec` for the StreamingC4 (Colossal Cleaned Common Crawl) dataset.

    Args:
        batch_size (int): Batch size per device.
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
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an
            exception. Default: 120 sec.
        version (int): Version of streaming (1 or 2). Default: 2.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """

    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if version == 1:
        dataset = StreamingC4(remote=remote,
                              local=local,
                              split=split,
                              shuffle=shuffle,
                              tokenizer_name=tokenizer_name,
                              max_seq_len=max_seq_len,
                              group_method=group_method,
                              max_retries=max_retries,
                              timeout=timeout,
                              batch_size=batch_size)

    elif version == 2:
        try:
            from streaming.text import C4
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e
        dataset = C4(tokenizer_name=tokenizer_name,
                     max_seq_len=max_seq_len,
                     group_method=group_method,
                     local=local,
                     remote=remote,
                     split=split,
                     shuffle=shuffle,
                     retry=max_retries,
                     timeout=timeout,
                     batch_size=batch_size)
    else:
        raise ValueError(f'Invalid streaming version: {version}')

    # Get collate_fn
    collate_fn = transformers.DataCollatorForLanguageModeling(tokenizer=dataset.tokenizer,
                                                              mlm=mlm,
                                                              mlm_probability=mlm_probability)

    return DataSpec(
        dataloader=DataLoader(
            dataset=dataset,  # type: ignore
            batch_size=batch_size,
            sampler=None,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **dataloader_kwargs),
        device_transforms=None)
