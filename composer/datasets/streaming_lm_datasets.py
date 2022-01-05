# Copyright 2021 MosaicML. All Rights Reserved.

import copy
import logging
import tempfile
from dataclasses import dataclass
from functools import partial
from itertools import chain
from os.path import join
from typing import List, Optional

import datasets
import torch
import yahp as hp
from transformers.testing_utils import CaptureLogger

from composer.core.types import Batch
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.utils import ddp
from composer.utils.data import get_subset_dataset

log = logging.getLogger(__name__)


def _split_dict_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if isinstance(batch, dict):
        chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
        num_chunks = len(list(chunked.values())[0])
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]
    else:
        raise ValueError(f'Expect batch from dataloader to be of type Dict[str, Tensor], but got {type(batch)}')


CACHED_DATASET_SIZES = {"c4": {"en": {"train": (1024, 356317), "validation": (8, 45576)}}}


@dataclass
class StreamingLMDatasetHparams(DatasetHparams):
    """
    Defines a generic dataset class for autoregressive and masked language models.
    """

    dataset_name: str = hp.optional("Name of the dataset to load.", default=None)
    dataset_config_name: Optional[str] = hp.optional(
        "If required, the specific configuration of the dataset that you would like to use.", default=None)
    split: str = hp.optional("What split of the dataset to use (e.g. 'train' or 'validation' or 'test')", default=None)
    tokenizer_name: str = hp.optional("The name of the tokenizer to preprocess text with.", default=None)
    max_seq_len: int = hp.optional("The max sequence length of each token sample.", default=None)
    group_method: str = hp.optional("How to group text samples into token samples.", default=None)
    use_masked_lm: bool = hp.optional("Whether the dataset shoud be encoded with masked language modeling or not.",
                                      default=None)
    mlm_probability: float = hp.optional("If using masked language modeling, the probability to mask tokens with.",
                                         default=0.15)
    seed: int = hp.optional("Which seed to use to generate train and validation splits.", default=5)
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        assert self.group_method in ["crop", "concat"], f"Unknown group_method: '{self.group_method}'"

        if self.use_masked_lm:
            if self.mlm_probability <= 0.0:
                raise ValueError(
                    "If using Masked Language Modeling, you must replace tokens with a non-zero probability.")
            raise NotImplementedError

    def load_dataset(self):
        return datasets.load_dataset(path=self.dataset_name,
                                     name=self.dataset_config_name,
                                     split=self.split,
                                     streaming=True)

    def get_approx_num_samples(self):
        try:
            n_shards, samples_per_shard = CACHED_DATASET_SIZES[self.dataset_name][self.dataset_config_name][self.split]
            return n_shards * samples_per_shard
        except:
            log.info("Iterating over dataset to determine # samples")
            dataset = self.load_dataset()
            samples = 0
            for _ in iter(dataset):
                samples += 1
            return samples

    def get_approx_num_tokens(self):
        return 1e12

    def _subsample(device_offset, text_batch):
        # Only return the i-th item out of N sequential items
        for k, v in text_batch.items():
            text_batch[k] = v[device_offset:device_offset + 1]
        return text_batch

    def _shard_dataset(self, dataset):
        # Select a subset of filepaths for sharded DDP training
        world_size = ddp.get_world_size()
        rank = ddp.get_global_rank()
        filepaths = dataset._ex_iterable.kwargs['filepaths']
        num_shards = len(filepaths)

        devices_per_shard = 1
        if world_size > num_shards:
            log.warning(
                f"Not enough unique shards ({num_shards}) for world size ({world_size}). Splitting shards among devices."
            )
            assert world_size % num_shards == 0, f"Cannot evenly split shards among devices"
            devices_per_shard = world_size // num_shards
        shard_offset = rank // devices_per_shard
        device_offset = rank % devices_per_shard

        device_filepaths = filepaths[shard_offset::world_size]
        dataset._ex_iterable.kwargs['filepaths'] = device_filepaths

        # Subsample dataset if shard is being shared among devices
        # NOTE: Mapping is executed in batched mode for better CPU utilization,
        # but the returned dataset is still an iterable over text samples
        if devices_per_shard > 1:
            dataset = dataset.map(
                partial(self._subsample, device_offset),
                batched=True,
                batch_size=devices_per_shard,
            )
        return dataset

    def _tokenize(text_batch):
        # Convert a text batch to a token batch
        return self.tokenizer(text_batch["text"])

    def _group_tokens(max_seq_len, group_method, token_batch):
        # Convert a batch of variable-length token samples to a grouped batch with fixed-length token samples
        # Every new token sample will have length 'max_seq_len'
        # Some tokens at the end of the original batch may be dropped.
        if group_method == "crop":
            raise NotImplementedError

        elif group_method == "concat":
            # Concatenate all tokens.
            concat_tokens = {}
            num_tokens = None
            for k, v in token_batch.items():
                concat_v = list(chain(*v))
                concat_tokens[k] = concat_v
                if num_tokens is None:
                    num_tokens = len(concat_v)
                else:
                    assert num_tokens == len(concat_v), "Not all values in concat_tokens dict have same len()"

            # We drop the small remainder of tokens at the end of the batch,
            # In the future we could support padding.
            if num_tokens >= max_seq_len:
                num_tokens = (num_tokens // max_seq_len) * max_seq_len

            # Split into token samples of size max_seq_len.
            result = {
                k: [v[i:i + max_seq_len] for i in range(0, num_tokens, max_seq_len)] for k, v in concat_tokens.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        else:
            raise ValueError(f"Unknown group_method: '{group_method}'")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataloaderSpec:
        assert dataloader_hparams.num_workers == 1, "LM Streaming Dataloader only supports num_workers=1"
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name, max_model_input_sizes=None)  #type: ignore (thirdparty)
        self.config = transformers.AutoConfig.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)

        # Load and shard dataset
        text_dataset = self.load_dataset()
        text_dataset = self.shard_dataset(text_dataset)

        # Shuffle
        if self.shuffle:
            text_dataset = text_dataset.shuffle(buffer_size=10000, seed=self.seed)

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
        grouped_token_dataset = token_dataset.map(
            partial(self._group_tokens, self.max_seq_len, self.group_method),
            batched=True,
            batch_size=token_sample_batch_size,
        )

        # Return DataloaderSpec
        return DataloaderSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=SizedIterableDataset(grouped_token_dataset),
            batch_size=batch_size,
            batch_sampler=None,
            drop_last=self.drop_last,
            collate_fn=transformers.data.data_collator.default_data_collator,
        ),
                              split_fn=_split_dict_fn)


class SizedIterableDataset(datasets.IterableDataset):

    def __init__(self, dataset):
        super().__init__(
            ex_iterable=dataset._ex_iterable,
            info=copy.deepcopy(dataset._info),
            split=dataset._split,
            format_type=dataset._format_type,
            shuffling=copy.deepcopy(dataset._shuffling),
        )

    def __len__(self):
        return 1e12  # HACK: iterable dataset size is not known
