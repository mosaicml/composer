# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, cast

from torch.utils.data import DataLoader, Dataset

from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder
from composer.utils import MissingConditionalImportError, dist

log = logging.getLogger(__name__)


def build_lm_dataloader(
    datadir: List[str],
    tokenizer_name: str,
    global_batch_size: int,
    *,
    split: str = 'train',
    shuffle: bool = True,
    drop_last: bool = True,
    use_masked_lm: bool = False,
    num_tokens: int = 0,
    mlm_probability: float = 0.15,
    subsample_ratio: float = 1.0,
    **dataloader_kwargs,
):
    """Builds a dataloader for a generic language modeling dataset.

    Args:
        datadir (list): List containing the string of the path to the HuggingFace
            Datasets directory.
        dataloader_hparams (DataLoaderHparams): DataLoaderHparams object.
        tokenizer_name (str): The name of the HuggingFace tokenizer to
            preprocess text with. See `HuggingFace documentation
            <https://huggingface.co/models>`_.
        global_batch_size (int): Global batch size.
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
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    try:
        import datasets
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='datasets') from e

    assert tokenizer_name is not None

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
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
        data_collator = transformers.default_data_collator
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                     mlm=use_masked_lm,
                                                                     mlm_probability=mlm_probability)

    sampler = dist.get_sampler(
        cast(Dataset, dataset),  # HF datasets do not subclass torch datasets, so this cast is needed
        drop_last=drop_last,
        shuffle=shuffle)

    return DataLoader(
        dataset=dataset,  # type: ignore
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=data_collator,
        **dataloader_kwargs)


def build_synthetic_lm_dataloader(
    synthetic_num_unique_samples: int,
    tokenizer_name: str,
    global_batch_size: int,
    *,
    split: str = 'train',
    shuffle: bool = True,
    drop_last: bool = True,
    use_masked_lm: bool = False,
    num_tokens: int = 0,
    mlm_probability: float = 0.15,
    subsample_ratio: float = 1.0,
    max_seq_length: int = 1024,
    **dataloader_kwargs,
):
    """Builds a synthetic dataloader for a generic language modeling dataset.

    Args:
        synthetic_num_unique_samples (int): Number of unique synthetic samples to generate.
        tokenizer_name (str): The name of the HuggingFace tokenizer to
            preprocess text with. See `HuggingFace documentation
            <https://huggingface.co/models>`_.
        global_batch_size (int)
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
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """

    try:
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    try:
        import datasets
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='datasets') from e

    assert tokenizer_name is not None

    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()

    column_names = ['text']

    # we just use the max sequence length in tokens to upper bound the sequence length in characters
    lm_datasets = synthetic_hf_dataset_builder(num_samples=synthetic_num_unique_samples,
                                               chars_per_sample=max_seq_length,
                                               column_names=column_names)

    tokenizer = generate_synthetic_tokenizer(tokenizer_family=tokenizer_name, dataset=lm_datasets)

    columns_to_remove = ['idx'] + column_names
    lm_datasets = lm_datasets.map(lambda inp: tokenizer(
        text=inp[column_names[0]], padding='max_length', max_length=max_seq_length, truncation=True),
                                  batched=True,
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
        data_collator = transformers.default_data_collator
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                     mlm=use_masked_lm,
                                                                     mlm_probability=mlm_probability)

    sampler = dist.get_sampler(
        cast(Dataset, dataset),  # HF datasets do not subclass torch datasets, so this cast is needed
        drop_last=drop_last,
        shuffle=shuffle)

    return DataLoader(
        dataset=dataset,  # type: ignore
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        collate_fn=data_collator,
        **dataloader_kwargs)
