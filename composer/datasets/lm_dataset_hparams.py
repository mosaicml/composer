# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Generic hyperparameters for self-supervised training of autoregressive and masked language models."""

import logging
from dataclasses import dataclass
from typing import List, Optional, cast

import yahp as hp
from torch.utils.data import DataLoader, Dataset

from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder
from composer.utils import MissingConditionalImportError, dist

__all__ = ['LMDatasetHparams']

log = logging.getLogger(__name__)


@dataclass
class LMDatasetHparams(DatasetHparams, SyntheticHparamsMixin):
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
        'Path to the Huggingface Datasets directory.', default_factory=list)
    split: Optional[str] = hp.optional("Whether to use 'train', 'validation' or 'test' split.", default=None)
    tokenizer_name: Optional[str] = hp.optional('The name of the tokenizer to preprocess text with.', default=None)
    use_masked_lm: bool = hp.optional('Whether the dataset should be encoded with masked language modeling or not.',
                                      default=False)
    num_tokens: int = hp.optional(doc='If desired, the number of tokens to truncate the dataset to.', default=0)
    mlm_probability: float = hp.optional('If using masked language modeling, the probability to mask tokens with.',
                                         default=0.15)
    seed: int = hp.optional('Which seed to use to generate train and validation splits.', default=5)
    subsample_ratio: float = hp.optional(default=1.0, doc='If desired, the percentage of the dataset to use.')
    max_seq_length: int = hp.optional(
        default=1024, doc='Optionally, the ability to set a custom sequence length for the training dataset.')

    def validate(self):
        if not self.use_synthetic:
            if self.datadir is None:
                raise ValueError('A data directory must be specified.')

        if self.tokenizer_name is None:
            raise ValueError('A tokenizer name must be specified to tokenize the dataset.')

        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("The dataset split must be one of 'train', 'validation', or 'test'.")

        if self.use_masked_lm is None:
            raise ValueError('To determine masking, use_masked_lm must be specified.')

        if self.use_masked_lm:
            if self.mlm_probability <= 0.0:
                raise ValueError(
                    'If using Masked Language Modeling, you must replace tokens with a non-zero probability.')

        if self.num_tokens > 0 and self.subsample_ratio < 1.0:
            raise Exception('Must specify one of num_tokens OR subsample_ratio, cannot specify both.')

        if (self.max_seq_length % 8 != 0):
            log.warning('For best hardware acceleration, it is recommended that sequence lengths be multiples of 8.')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        self.validate()

        if self.use_synthetic:
            return build_synthetic_lm_dataloader(
                synthetic_num_unique_samples=self.synthetic_num_unique_samples,
                dataloader_hparams=dataloader_hparams,
                tokenizer_name=self.tokenizer_name,  # type: ignore
                batch_size=batch_size,
                split=self.split,  # type: ignore
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                use_masked_lm=self.use_masked_lm,
                num_tokens=self.num_tokens,
                mlm_probability=self.mlm_probability,
                subsample_ratio=self.subsample_ratio)
        else:
            return build_lm_dataloader(
                datadir=self.datadir,
                dataloader_hparams=dataloader_hparams,
                tokenizer_name=self.tokenizer_name,  # type: ignore
                batch_size=batch_size,
                split=self.split,  # type: ignore
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                use_masked_lm=self.use_masked_lm,
                num_tokens=self.num_tokens,
                mlm_probability=self.mlm_probability,
                subsample_ratio=self.subsample_ratio)


def build_lm_dataloader(
    datadir: List[str],
    dataloader_hparams: DataLoaderHparams,
    tokenizer_name: str,
    batch_size: int,
    *,
    split: str = 'train',
    shuffle: bool = True,
    drop_last: bool = True,
    use_masked_lm: bool = False,
    num_tokens: int = 0,
    mlm_probability: float = 0.15,
    subsample_ratio: float = 1.0,
):
    """Builds a dataloader for a generic language modeling dataset.

    Args:
        datadir (list): List containing the string of the path to the HuggingFace
            Datasets directory.
        dataloader_hparams (DataLoaderHparams): DataLoaderHparams object.
        tokenizer_name (str): The name of the HuggingFace tokenizer to
            preprocess text with. See `HuggingFace documentation
            <https://huggingface.co/models>`_.
        batch_size (int): Batch size per device.
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default: ``'train```. Default: ``'train'``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        use_masked_lm (bool): Whether the dataset should be encoded with masked
            language modeling or not.
        num_tokens (int, optional): Number of tokens to train on. ``0``
            will train on all tokens in the dataset. Default: ``0``.
        mlm_probability (float, optional): If using masked language modeling, the
            probability with which tokens will be masked. Default: ``0.15``.
        subsample_ratio (float, optional): Proportion of the dataset to use. Default:
            ``1.0``.
    """

    try:
        import datasets  # type: ignore
        import transformers  # type: ignore
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    assert tokenizer_name is not None

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)  #type: ignore (thirdparty)
    # loads a dataset that is assumed to be pre-tokenized
    lm_datasets = [datasets.load_from_disk(i) for i in datadir]  #type: ignore (thirdparty)

    # merge the dataset to re-sample from
    if split is None:
        raise ValueError('A dataset split is required')
    merged_dataset = [[d[split]] for d in lm_datasets]
    # flatten merged_dataset
    merged_dataset = [item for sublist in merged_dataset for item in sublist]
    lm_datasets = datasets.concatenate_datasets(merged_dataset)  #type: ignore (thirdparty)

    total_num_samples = len(lm_datasets)  # type: ignore
    tokens_per_sample = len(lm_datasets[0]['input_ids'])  #type: ignore (thirdparty)
    total_num_tokens = total_num_samples * tokens_per_sample

    # truncate the dataset to a specified size
    num_samples = total_num_samples
    if num_tokens > 0:
        assert num_tokens <= total_num_tokens, f'Requested {num_tokens} tokens must be <= total_num_tokens={total_num_tokens}'
        assert num_tokens % tokens_per_sample == 0, f'Requested {num_tokens} tokens is not divisible by tokens_per_sample={tokens_per_sample}'
        num_samples = num_tokens // tokens_per_sample
        subsample_ratio = num_samples / total_num_samples
    elif subsample_ratio < 1.0:
        num_samples = round(total_num_samples * subsample_ratio)
        num_tokens = num_samples * tokens_per_sample
    elif subsample_ratio == 1.0 and num_tokens == 0:
        num_tokens = total_num_tokens
    else:
        log.warning('No subsampling going on!')

    lm_datasets = lm_datasets.select(range(num_samples))  # type: ignore (thirdparty)
    log.info(f'LM datasets: {lm_datasets}')
    log.info(f'Subsample ratio: {subsample_ratio}')
    log.info(f'Total number of samples: {num_samples:e}')
    log.info(f'Total number of tokens: {num_tokens:e}')
    dataset = lm_datasets

    # for some tokenizers, e.g. GPT-2, they don't have padding tokens. Hence, we cannot use the LM collator.
    if tokenizer.pad_token_id is None:
        data_collator = transformers.default_data_collator  # type: ignore
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(  # type: ignore
            tokenizer=tokenizer, mlm=use_masked_lm, mlm_probability=mlm_probability)

    sampler = dist.get_sampler(
        cast(Dataset, dataset),  # HF datasets do not subclass torch datasets, so this cast is needed
        drop_last=drop_last,
        shuffle=shuffle)

    return dataloader_hparams.initialize_object(
        dataset=dataset,  # type: ignore
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=data_collator)


def build_synthetic_lm_dataloader(
    synthetic_num_unique_samples: int,
    dataloader_hparams: DataLoaderHparams,
    tokenizer_name: str,
    batch_size: int,
    *,
    split: str = 'train',
    shuffle: bool = True,
    drop_last: bool = True,
    use_masked_lm: bool = False,
    num_tokens: int = 0,
    mlm_probability: float = 0.15,
    subsample_ratio: float = 1.0,
    max_seq_length: int = 1024,
):
    """Builds a synthetic dataloader for a generic language modeling dataset.

    Args:
        synthetic_num_unique_samples (int): Number of unique synthetic samples to generate.
        dataloader_hparams (DataLoaderHparams): DataLoaderHparams object.
            Datasets directory.
        tokenizer_name (str): The name of the HuggingFace tokenizer to
            preprocess text with. See `HuggingFace documentation
            <https://huggingface.co/models>`_.
        batch_size (int): Batch size per device.
        split (str): the dataset split to use either 'train', 'val', or 'test'. Default:
        ``'train```. Default: ``'train'``.
        shuffle (bool): whether to shuffle the dataset. Default: ``True``.
        drop_last (bool): whether to drop last samples. Default: ``True``.
        use_masked_lm (bool): Whether the dataset should be encoded with masked
            language modeling or not.
        num_tokens (int, optional): Number of tokens to train on. ``0``
            will train on all tokens in the dataset. Default: ``0``.
        mlm_probability (float, optional): If using masked language modeling, the
            probability with which tokens will be masked. Default: ``0.15``.
        subsample_ratio (float, optional): Proportion of the dataset to use. Default:
            ``1.0``.
        max_seq_length (int, optional): Maximum sequence length for datasets.
            Default: ``1024``.
    """

    try:
        import datasets  # type: ignore
        import transformers  # type: ignore
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    assert tokenizer_name is not None

    column_names = ['text']

    # we just use the max sequence length in tokens to upper bound the sequence length in characters
    lm_datasets = synthetic_hf_dataset_builder(num_samples=synthetic_num_unique_samples,
                                               chars_per_sample=max_seq_length,
                                               column_names=column_names)

    tokenizer = generate_synthetic_tokenizer(tokenizer_family=tokenizer_name, dataset=lm_datasets)

    columns_to_remove = ['idx'] + column_names
    lm_datasets = lm_datasets.map(
        lambda inp: tokenizer(
            text=inp[column_names[0]], padding='max_length', max_length=max_seq_length, truncation=True),
        batched=True,
        num_proc=None if dataloader_hparams.num_workers == 0 else dataloader_hparams.num_workers,
        remove_columns=columns_to_remove,
        keep_in_memory=True)

    # override sizing to able use of synthetic datasets
    num_tokens = 0
    subsample_ratio = 1.0
    lm_datasets = [{split: lm_datasets}]

    # merge the dataset to re-sample from
    if split is None:
        raise ValueError('A dataset split is required')
    merged_dataset = [[d[split]] for d in lm_datasets]
    # flatten merged_dataset
    merged_dataset = [item for sublist in merged_dataset for item in sublist]
    lm_datasets = datasets.concatenate_datasets(merged_dataset)  #type: ignore (thirdparty)

    total_num_samples = len(lm_datasets)  # type: ignore
    tokens_per_sample = len(lm_datasets[0]['input_ids'])  #type: ignore (thirdparty)
    total_num_tokens = total_num_samples * tokens_per_sample

    # truncate the dataset to a specified size
    num_samples = total_num_samples
    if num_tokens > 0:
        assert num_tokens <= total_num_tokens, f'Requested {num_tokens} tokens must be <= total_num_tokens={total_num_tokens}'
        assert num_tokens % tokens_per_sample == 0, f'Requested {num_tokens} tokens is not divisible by tokens_per_sample={tokens_per_sample}'
        num_samples = num_tokens // tokens_per_sample
        subsample_ratio = num_samples / total_num_samples
    elif subsample_ratio < 1.0:
        num_samples = round(total_num_samples * subsample_ratio)
        num_tokens = num_samples * tokens_per_sample
    elif subsample_ratio == 1.0 and num_tokens == 0:
        num_tokens = total_num_tokens
    else:
        log.warning('No subsampling going on!')

    lm_datasets = lm_datasets.select(range(num_samples))  # type: ignore (thirdparty)
    log.info(f'LM datasets: {lm_datasets}')
    log.info(f'Subsample ratio: {subsample_ratio}')
    log.info(f'Total number of samples: {num_samples:e}')
    log.info(f'Total number of tokens: {num_tokens:e}')
    dataset = lm_datasets

    # for some tokenizers, e.g. GPT-2, they don't have padding tokens. Hence, we cannot use the LM collator.
    if tokenizer.pad_token_id is None:
        data_collator = transformers.default_data_collator  # type: ignore
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(  # type: ignore
            tokenizer=tokenizer, mlm=use_masked_lm, mlm_probability=mlm_probability)

    sampler = dist.get_sampler(
        cast(Dataset, dataset),  # HF datasets do not subclass torch datasets, so this cast is needed
        drop_last=drop_last,
        shuffle=shuffle)

    return dataloader_hparams.initialize_object(
        dataset=dataset,  # type: ignore
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=data_collator)
