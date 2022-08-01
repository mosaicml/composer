# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""C4 (Colossal Cleaned Common Crawl) dataset.

This dataset is a colossal, cleaned version of Common Crawl's web crawl corpus and it is based on the `Common Crawl
<https://commoncrawl.org>`_ dataset.
"""
import copy
import logging
import os
from functools import partial
from itertools import chain, cycle
from typing import Any, Dict, Optional

from torch.utils.data import IterableDataset, get_worker_info

from composer.datasets.streaming import StreamingDataset
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['C4Dataset', 'StreamingC4']


class StreamingC4(StreamingDataset):
    """
    Implementation of the C4 (Colossal Cleaned Common Crawl) dataset using StreamingDataset.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Currently only supporting ``'truncate'``.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def _decode(self, data: bytes) -> str:
        return data.decode('utf-8')

    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        else:
            truncation = False
            padding = False
            max_length = None
        return self.tokenizer(text_sample['text'], truncation=truncation, padding=padding, max_length=max_length)

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

        # HF Transformers is needed to build the tokenizer
        try:
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

        # Validation
        if split not in ['train', 'val']:
            raise ValueError(f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate']:
            raise ValueError(f"Only group_method='truncate' is supported at this time.")

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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        # Skip any token grouping, currently only supporting group_method='truncate'
        return token_sample


class C4Dataset(IterableDataset):
    """Builds a streaming, sharded, sized :class:`torch.utils.data.IterableDataset` for the C4 (Colossal Cleaned
    Common Crawl) dataset. Used for pretraining autoregressive or masked language models. Text samples are streamed
    directly from the cloud using HuggingFace's C4 Dataset with streaming backend (See
    https://huggingface.co/datasets/c4 for more details). The text samples are then shuffled, tokenized, and grouped on-
    the-fly.

    Args:
        split (str): What split of the dataset to use. Either ``'train'`` or ``'validation'``.
        num_samples (int): The number of post-processed token samples, used to set epoch size of the :class:`torch.data.utils.IterableDataset`.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text with.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Either ``'truncate'`` or ``'concat'``.
        shuffle (bool): Whether to shuffle the samples in the dataset. Currently, shards are assigned and consumed with
            deterministic per-device shard order, but shuffling affects the order of samples via (per-device) shuffle
            buffers. Default: ``False``.
        shuffle_buffer_size (int): If ``shuffle=True``, samples are read into a buffer of this size (per-device), and
            randomly sampled from there to produce shuffled samples. Default: ``10000``.
        seed (int): If ``shuffle=True``, what seed to use for shuffling operations. Default: ``5``.
    Returns:
        IterableDataset: A :class:`torch.utils.data.IterableDataset` object.
    """

    def __init__(self,
                 split,
                 num_samples,
                 tokenizer_name,
                 max_seq_len,
                 group_method,
                 shuffle=False,
                 shuffle_buffer_size=10000,
                 seed=5):
        try:
            import datasets
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='datasets transformers') from e

        self.split = split
        self.num_samples = num_samples
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        # Metadata
        c4_metadata = {
            'train': {
                'num_shards': 1024,
                'approx_samples_per_shard': 356317,
            },
            'validation': {
                'num_shards': 8,
                'approx_samples_per_shard': 45576,
            }
        }
        if self.split in c4_metadata:
            self.num_shards = c4_metadata[self.split]['num_shards']
            self.approx_samples_per_shard = c4_metadata[self.split]['approx_samples_per_shard']
        else:
            raise ValueError(f'Unknown split={self.split}, expected one of {list(c4_metadata.keys())}.')

        # Set dataset size
        self.world_size = dist.get_world_size()
        self.rank = dist.get_global_rank()
        self.num_samples_per_device = self.num_samples // self.world_size
        if self.num_samples % self.world_size != 0:
            new_num_samples = self.num_samples_per_device * self.world_size
            log.warning(
                f'Num samples will be truncated from {num_samples}->{new_num_samples} to maintain divisibility across {self.world_size} devices.'
            )
            self.num_samples = new_num_samples

        # Try and detect if num_samples is larger than original dataset
        original_approx_samples = self.num_shards * self.approx_samples_per_shard
        if self.num_samples > original_approx_samples and self.group_method == 'truncate':
            log.warning(
                f"Num samples was set to {self.num_samples} with group_method 'truncate' but split '{split}' has only {original_approx_samples}. "
                f'The original dataset will cycle until the new nominal length of {self.num_samples}.')
        if self.group_method == 'concat':
            log.warning(
                f"When using group_method 'concat', sequential token samples are concatenated and chunked into fixed-length samples of size max_seq_len={self.max_seq_len}. "
                f'In general we cannot detect ahead-of-time if your setting of num_samples={self.num_samples} will be larger than the original dataset, '
                f'but if it is larger, the original dataset will cycle until the new nominal length of {self.num_samples}.'
            )

        # Build tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load and shard dataset
        text_dataset = datasets.load_dataset(path='c4', name='en', split=split, streaming=True)
        text_dataset = self._shard_dataset(text_dataset)
        if not isinstance(text_dataset, datasets.IterableDataset):
            raise ValueError('Unable to build sharded Huggingface C4 Dataset.')

        # Map text samples to token samples
        # NOTE: Mapping is executed in batched mode for better CPU utilization,
        # but the returned dataset is still an iterable over tokenized samples
        text_sample_batch_size = 1000
        token_dataset = text_dataset.map(
            self._tokenize,
            batched=True,
            batch_size=text_sample_batch_size,
        )

        # Map variable-length token samples to fixed-length token samples
        # NOTE: Mapping is executed in batched mode for better CPU utilization,
        # but the returned dataset is still an iterable over tokenized samples.
        # NOTE: Depending on the 'group_method', this step may alter the number of
        # token samples in the dataset, and may mix neighboring token samples together.
        token_sample_batch_size = 1000
        token_dataset = token_dataset.map(
            self._group_tokens,
            batched=True,
            batch_size=token_sample_batch_size,
        )

        # Repeat over the dataset
        # TODO: This functionality should eventually be upstreamed to HF as `hf_iterable_dataset.repeat()`
        repeat_token_dataset = self._repeat(token_dataset)

        # Limit the number of post-processed token samples
        sized_token_dataset = repeat_token_dataset.take(self.num_samples_per_device)

        # Shuffle post-processed token samples
        # Samples are read into and randomly sampled from per-device shuffle buffer
        if self.shuffle:
            sized_token_dataset = sized_token_dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed)

        # Finish
        self.iterable_dataset = sized_token_dataset

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers != 1:
            raise ValueError("Multi-worker processing not supported for this dataset yet, please use 'num_workers=1'.")
        return iter(self.iterable_dataset)

    def __len__(self):
        return self.num_samples_per_device

    # Repeat a HF iterable dataset infinitely
    # TODO: This functionality should eventually be upstreamed to HF as `hf_iterable_dataset.repeat()`
    def _repeat(self, dataset):
        try:
            from datasets.iterable_dataset import _BaseExamplesIterable
            from datasets.iterable_dataset import iterable_dataset as hf_iterable_dataset
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='datasets') from e

        class RepeatExamplesIterable(_BaseExamplesIterable):

            def __init__(self, ex_iterable):
                self.ex_iterable = ex_iterable

            def __iter__(self):
                yield from cycle(self.ex_iterable)

            @property
            def n_shards(self):
                return self.ex_iterable.n_shards

        ex_iterable = RepeatExamplesIterable(dataset._ex_iterable)
        return hf_iterable_dataset(
            ex_iterable=ex_iterable,
            info=dataset._info.copy(),
            split=dataset._split,
            format_type=dataset._format_type,
            shuffling=copy.deepcopy(dataset._shuffling),
        )

    def _subsample(self, device_offset, text_batch):
        # Only return the i-th item out of N sequential items
        for k, v in text_batch.items():
            text_batch[k] = v[device_offset:device_offset + 1]
        return text_batch

    # Take a HF iterable dataset with multiple shards and prepare it for data-parallel training
    # Shards are split per-device, e.g. For 8 shards and 4 devices... device0 receives shards [0, 4], device1 receives shards [1, 5].. etc.
    # If there are not enough shards for devices (common with small validation splits), then shards are sent to multiple devices but subsampled internally.
    # E.g. For 2 shards and 4 devices... device0 receives shards [0] and consumes samples 0, 2, 4, ... device1 recieves shards [0] and consumes samples 1, 3, 5, ... etc.
    # Currently, either (num_shards % num_devices == 0) or (num_devices % num_shards == 0) is enforced for efficient streaming,
    # but this could be relaxed in the future at the cost of increased bandwidth (have many more devices read the same shards and subsample)
    def _shard_dataset(self, dataset):
        # Verify # of shards
        filepaths = dataset._ex_iterable.kwargs['filepaths']
        if self.num_shards != len(filepaths):
            raise ValueError(f'Found {len(filepaths)} shards, expected {self.num_shards}')

        # Determine how to allocate devices to shards
        devices_per_shard = 1
        if self.num_shards < self.world_size:
            log.warning(
                f'Not enough unique shards ({self.num_shards}) for world size ({self.world_size}). Splitting shards among devices.'
            )
            if self.world_size % self.num_shards != 0:
                raise ValueError(f'Cannot evenly split {self.num_shards} shards among {self.world_size} devices')
            devices_per_shard = self.world_size // self.num_shards
        elif self.num_shards % self.world_size != 0:
            raise ValueError(f'Cannot evenly split {self.num_shards} shards among {self.world_size} devices')
        shard_offset = self.rank // devices_per_shard
        device_offset = self.rank % devices_per_shard

        # Select a deterministic subset of shards
        device_filepaths = filepaths[shard_offset::self.world_size]
        dataset._ex_iterable.kwargs['filepaths'] = device_filepaths

        # Subsample shard if shard is being shared among devices
        # NOTE: Mapping is executed in batched mode for better CPU utilization,
        # but the returned dataset is still an iterable over text samples
        if devices_per_shard > 1:
            dataset = dataset.map(
                partial(self._subsample, device_offset),
                batched=True,
                batch_size=devices_per_shard,
            )
        return dataset

    # Use the initialized HF tokenizer to convert a text batch to a token batch
    def _tokenize(self, text_batch):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        else:
            truncation = False
            padding = False
            max_length = None
        return self.tokenizer(text_batch['text'], truncation=truncation, padding=padding, max_length=max_length)

    # Prepare a batch of token samples for pretraining, by grouping them into fixed-length token samples, with either 'truncate' or 'concat' group methods.
    # If using 'truncate', each token sample is padded/truncated to 'self.max_seq_len', and there is no mixing between adjacent token samples.
    # Using 'truncate' may be computationally inefficient if 'self.max_seq_len' is large, as the suffix of each token sample will consist of empty padding.
    # Using 'truncate' may also be data inefficent as it will discard the suffix of any token sample that is larger than 'self.max_seq_len'.
    # If using 'concat', the batch of token samples is concatenated and chunked, such that every new token sample is exactly 'self.max_seq_len' long with no padding.
    # Using 'concat' may drop a small amount of data at the end of each batch if the total number of tokens is not divisible by 'self.max_seq_len'.
    # Using 'concat' will alter the number of token samples in the iterable dataset, and differently per-device,
    # so we require the user to provide a 'self.num_samples' limit to ensure epoch-boundary synchronization across devices.
    def _group_tokens(self, token_batch):
        if self.group_method == 'truncate':
            # No processing needed, as 'self._tokenize()' has already padded / truncated each token sample to 'self.max_seq_len'
            return token_batch
        elif self.group_method == 'concat':
            # Concatenate all tokens.
            concat_tokens = {}
            num_tokens = None
            for k, v in token_batch.items():
                concat_v = list(chain(*v))
                concat_tokens[k] = concat_v
                if num_tokens is None:
                    num_tokens = len(concat_v)
                elif num_tokens != len(concat_v):
                    raise ValueError('Not all values in concat_tokens dict have same len()')
                else:
                    pass
            if num_tokens is None:
                raise ValueError('Failed to determine num_tokens.')

            # We drop the small remainder of tokens at the end of the batch.
            if num_tokens >= self.max_seq_len:
                num_tokens = (num_tokens // self.max_seq_len) * self.max_seq_len

            # Split into token samples of size max_seq_len.
            result = {
                k: [v[i:i + self.max_seq_len] for i in range(0, num_tokens, self.max_seq_len)
                   ] for k, v in concat_tokens.items()
            }
            return result
        else:
            raise ValueError(f"Unknown group_method: '{self.group_method}'")
