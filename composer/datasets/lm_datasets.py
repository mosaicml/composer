# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import tempfile
from dataclasses import dataclass
from os.path import join
from typing import List

import yahp as hp

from composer.core.types import Batch
from composer.datasets.hparams import DataloaderSpec, DatasetHparams

log = logging.getLogger(__name__)


def _split_dict_fn(batch: Batch, n_microbatches: int) -> List[Batch]:
    if isinstance(batch, dict):
        chunked = {k: v.chunk(n_microbatches) for k, v in batch.items()}
        num_chunks = len(list(chunked.values())[0])
        return [{k: v[idx] for k, v in chunked.items()} for idx in range(num_chunks)]
    else:
        raise ValueError(f'Expect batch from dataloader to be of type Dict[str, Tensor], but got {type(batch)}')


@dataclass
class LMDatasetHparams(DatasetHparams):
    """
    Defines a generic dataset class for autoregressive language models.
    """

    datadir: List[str] = hp.required("Path to the Huggingface Datasets directory.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    num_tokens: int = hp.optional(doc='If desired, the number of tokens to truncate the dataset to.', default=0)
    seed: int = hp.optional("Which seed to use to generate train and validation splits.", default=5)
    subsample_ratio: float = hp.optional(default=1.0, doc='If desired, the percentage of the dataset to use.')
    train_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    val_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the validation dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def initialize_object(self) -> DataloaderSpec:
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.config = transformers.AutoConfig.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        lm_datasets = [datasets.load_from_disk(i) for i in self.datadir]  #type: ignore (thirdparty)

        # TODO: this re-loads a large dataset into memory three times
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("The dataset split must be one of 'train', 'validation', or 'test'.")

        # merge the dataset to re-sample from
        merged_dataset = [[d[self.split]] for d in lm_datasets]
        # flatten merged_dataset
        merged_dataset = [item for sublist in merged_dataset for item in sublist]
        lm_datasets = datasets.concatenate_datasets(merged_dataset)  #type: ignore (thirdparty)

        # generate a cache file name so the training and validation set use the same split
        indices_cache_file_name = join(tempfile.gettempdir(), f"{self.seed}.indicies")

        # shuffle the dataset
        lm_datasets = lm_datasets.shuffle(indices_cache_file_name=indices_cache_file_name, seed=self.seed)

        if self.num_tokens > 0 and self.subsample_ratio < 1.0:
            raise Exception("Must specify one of num_tokens OR subsample_ratio, cannot specify both.")

        total_num_samples = len(lm_datasets)
        tokens_per_sample = len(lm_datasets[0]['input_ids'])
        total_num_tokens = total_num_samples * tokens_per_sample

        # truncate the dataset to a specified size
        num_samples = total_num_samples
        if self.num_tokens > 0:
            assert self.num_tokens <= total_num_tokens, f"Requested {self.num_tokens} tokens must be <= total_num_tokens={total_num_tokens}"
            assert self.num_tokens % tokens_per_sample == 0, f"Requested {self.num_tokens} tokens is not divisible by tokens_per_sample={tokens_per_sample}"
            num_samples = self.num_tokens // tokens_per_sample
            self.subsample_ratio = num_samples / total_num_samples
        elif self.subsample_ratio < 1.0:
            num_samples = round(total_num_samples * self.subsample_ratio)
            self.num_tokens = num_samples * tokens_per_sample
        else:
            log.warning("No subsampling going on!")

        lm_datasets = lm_datasets.select(range(num_samples))
        log.info(f"LM datasets: {lm_datasets}")
        log.info(f"Subsample ratio: {self.subsample_ratio}")
        log.info(f"Total number of samples: {num_samples:e}")
        log.info(f"Total number of tokens: {self.num_tokens:e}")
        self.dataset = lm_datasets

        self.data_collator = transformers.default_data_collator

        return DataloaderSpec(
            dataset=self.dataset,  #type: ignore (thirdparty)
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn)
