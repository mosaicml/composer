# Copyright 2021 MosaicML. All Rights Reserved.

"""Generic dataset class for self-supervised training of autoregressive and masked language models."""

import logging
import tempfile
import textwrap
from dataclasses import dataclass
from os.path import join
from typing import List, Optional

import yahp as hp

from composer.core.types import Batch, DataSpec
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.utils import dist

__all__ = ["LMDatasetHparams"]

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
    """Defines a generic dataset class for self-supervised training of autoregressive and masked language models.

    Args:
        datadir (list): List containing the string of the path to the HuggingFace
            Datasets directory.
        split (str): Whether to use ``'train'``, ``'test'``, or
            ``'validation'`` split.
        tokenizer_name (str): The name of the HuggingFace tokenizer to
            preprocess text with. See `HuggingFace documentation
            <https://huggingface.co/models>`_.
        use_masked_lm (bool): Whether the dataset should be encoded with masked
            language modeling or not.
        num_tokens (int, optional): Number of tokens to train on. ``0``
            will train on all tokens in the dataset. Default: ``0``.
        mlm_probability (float, optional): If using masked language modeling, the
            probability with which tokens will be masked. Default: ``0.15``.
        seed (int, optional): Random seed for generating train and validation splits.
            Default: ``5``.
        subsample_ratio (float, optional): Proportion of the dataset to use. Default:
            ``1.0``.
        train_sequence_length (int, optional): Sequence length for training dataset.
            Default: ``1024``.
        val_sequence_length (int, optional): Sequence length for validation dataset.
            Default: ``1024``.
    """

    # TODO(moin): Switch datadir to be a string, rather than a list of strings, to be similar to the
    # other datasets
    datadir: List[str] = hp.optional(  # type: ignore
        "Path to the Huggingface Datasets directory.", default_factory=list)
    split: Optional[str] = hp.optional("Whether to use 'train', 'validation' or 'test' split.", default=None)
    tokenizer_name: Optional[str] = hp.optional("The name of the tokenizer to preprocess text with.", default=None)
    use_masked_lm: bool = hp.optional("Whether the dataset should be encoded with masked language modeling or not.",
                                      default=None)
    num_tokens: int = hp.optional(doc='If desired, the number of tokens to truncate the dataset to.', default=0)
    mlm_probability: float = hp.optional("If using masked language modeling, the probability to mask tokens with.",
                                         default=0.15)
    seed: int = hp.optional("Which seed to use to generate train and validation splits.", default=5)
    subsample_ratio: float = hp.optional(default=1.0, doc='If desired, the percentage of the dataset to use.')
    train_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    val_sequence_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the validation dataset.')

    def validate(self):
        if self.datadir is None:
            raise ValueError("A data directory must be specified.")

        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("The dataset split must be one of 'train', 'validation', or 'test'.")

        if self.tokenizer_name is None:
            raise ValueError("A tokenizer name must be specified to tokenize the dataset.")

        if self.use_masked_lm is None:
            raise ValueError("To determine masking, use_masked_lm must be specified.")

        if self.use_masked_lm:
            if self.mlm_probability <= 0.0:
                raise ValueError(
                    "If using Masked Language Modeling, you must replace tokens with a non-zero probability.")

        if self.num_tokens > 0 and self.subsample_ratio < 1.0:
            raise Exception("Must specify one of num_tokens OR subsample_ratio, cannot specify both.")

        if (self.train_sequence_length % 8 != 0) or (self.val_sequence_length % 8 != 0):
            log.warning("For best hardware acceleration, it is recommended that sequence lengths be multiples of 8.")

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataSpec:
        try:
            import datasets
            import transformers
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
                if using pip or `conda install -c conda-forge datasets transformers` if using Anaconda.""")) from e

        self.validate()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.config = transformers.AutoConfig.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        lm_datasets = [datasets.load_from_disk(i) for i in self.datadir]  #type: ignore (thirdparty)

        # merge the dataset to re-sample from
        if self.split is None:
            raise ValueError("split is required")
        merged_dataset = [[d[self.split]] for d in lm_datasets]
        # flatten merged_dataset
        merged_dataset = [item for sublist in merged_dataset for item in sublist]
        lm_datasets = datasets.concatenate_datasets(merged_dataset)  #type: ignore (thirdparty)

        # generate a cache file name so the training and validation set use the same split
        indices_cache_file_name = join(tempfile.gettempdir(), f"{self.seed}.indices")

        # shuffle the dataset
        lm_datasets = lm_datasets.shuffle(indices_cache_file_name=indices_cache_file_name, seed=self.seed)

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
        elif self.subsample_ratio == 1.0 and self.num_tokens == 0:
            self.num_tokens = total_num_tokens
        else:
            log.warning("No subsampling going on!")

        lm_datasets = lm_datasets.select(range(num_samples))
        log.info(f"LM datasets: {lm_datasets}")
        log.info(f"Subsample ratio: {self.subsample_ratio}")
        log.info(f"Total number of samples: {num_samples:e}")
        log.info(f"Total number of tokens: {self.num_tokens:e}")
        dataset = lm_datasets

        # for some tokenizers, e.g. GPT-2, they don't have padding tokens. Hence, we cannot use the LM collator.
        if self.tokenizer.pad_token_id is None:
            data_collator = transformers.default_data_collator
        else:
            data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                                         mlm=self.use_masked_lm,
                                                                         mlm_probability=self.mlm_probability)

        sampler = dist.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)

        return DataSpec(dataloader=dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=data_collator,
        ),
                        split_batch=_split_dict_fn)
