# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic language modeling datasets used for testing, profiling, and debugging."""

from __future__ import annotations

import json
import string
from os.path import join
from random import Random
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, NamedTuple, Optional

from composer.utils import MissingConditionalImportError

if TYPE_CHECKING:
    import tokenizers.models as tokenizers_models
    from datasets import Dataset
    from tokenizers import decoders, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizer

__all__ = ['SyntheticTokenizerParams', 'generate_synthetic_tokenizer', 'synthetic_hf_dataset_builder']


class SyntheticTokenizerParams(NamedTuple):
    tokenizer_model: tokenizers_models.Model
    normalizer: normalizers.Normalizer
    pre_tokenizer: pre_tokenizers.PreTokenizer
    decoder: decoders.Decoder
    initial_alphabet: list
    special_tokens: list
    pad_token: str
    trainer_cls: type
    tokenizer_cls: type


def _generate_bert_tokenizer_params(dataset) -> SyntheticTokenizerParams:
    try:
        import tokenizers.models as tokenizers_models
        import tokenizers.trainers as tokenizers_trainer
        from tokenizers import decoders, normalizers, pre_tokenizers
        from transformers import BertTokenizer
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    unk_token = '[UNK]'
    pad_token = '[PAD]'

    initial_alphabet = ''.join([i for i in dataset])
    initial_alphabet = list(set(initial_alphabet))

    return SyntheticTokenizerParams(
        tokenizer_model=tokenizers_models.WordPiece(unk_token=unk_token),  # type: ignore
        normalizer=normalizers.BertNormalizer(),
        pre_tokenizer=pre_tokenizers.BertPreTokenizer(),
        decoder=decoders.WordPiece(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[pad_token, unk_token, '[SEP]', '[CLS]', '[MASK]'],
        pad_token=pad_token,
        trainer_cls=tokenizers_trainer.WordPieceTrainer,
        tokenizer_cls=BertTokenizer,
    )


def _generate_gpt2_tokenizer_params() -> SyntheticTokenizerParams:
    try:
        import tokenizers.models as tokenizers_models
        import tokenizers.trainers as tokenizers_trainer
        from tokenizers import decoders, normalizers, pre_tokenizers
        from transformers import GPT2Tokenizer
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    unk_token = None
    pad_token = '<pad>'

    return SyntheticTokenizerParams(
        tokenizer_model=tokenizers_models.BPE(unk_token=unk_token),
        normalizer=normalizers.Lowercase(),
        pre_tokenizer=pre_tokenizers.ByteLevel(),
        decoder=decoders.ByteLevel(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[pad_token, '<endoftext>'],
        pad_token=pad_token,
        trainer_cls=tokenizers_trainer.BpeTrainer,
        tokenizer_cls=GPT2Tokenizer,
    )


def generate_synthetic_tokenizer(tokenizer_family: str,
                                 dataset: Optional[Dataset] = None,
                                 vocab_size: int = 256) -> PreTrainedTokenizer:
    """Generates a synthetic tokenizer based on a tokenizer family.

    Args:
        tokenizer_family (str): Which tokenizer family to emulate. One of ['gpt2', 'bert'].
        dataset (Optional[datasets.Dataset]): Optionally, the dataset to train the tokenzier off of.
            If ``None``, a :class:`~SyntheticHFDataset` will be generated. Default: ``None``.
        vocab_size (int): The size of the tokenizer vocabulary. Defaults to 256.
    """

    try:
        import tokenizers.models as tokenizers_models
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizer
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    # generate a synthetic dataset with reasonable defaults is none is provided
    if dataset is None:
        num_samples = 100
        chars_per_sample = 128
        column_names = ['sentence']
        dataset = synthetic_hf_dataset_builder(num_samples=num_samples,
                                               chars_per_sample=chars_per_sample,
                                               column_names=column_names)

    # change a columnar dataset into a list
    flattened_columns = [dataset[key] for key in dataset.column_names if key != 'idx']
    # flatten the list of lists into a single list
    flattened_dataset = []
    for sublist in flattened_columns:
        for item in sublist:
            flattened_dataset.append(item)

    if 'bert' in tokenizer_family:
        tokenizer_params = _generate_bert_tokenizer_params(flattened_dataset)
    elif 'gpt2' in tokenizer_family:
        tokenizer_params = _generate_gpt2_tokenizer_params()
    else:
        raise ValueError(f'Synthetic tokenizers for tokenizer family {tokenizer_family} are currently unsupported.')

    tokenizer = Tokenizer(tokenizer_params.tokenizer_model)
    tokenizer.enable_padding(direction='right',
                             pad_id=0,
                             pad_type_id=0,
                             pad_token=tokenizer_params.pad_token,
                             pad_to_multiple_of=8)
    # The 'type: ignore' is because the underlying Rust package has improper type annotations. PyRight throws:
    # Cannot assign member "normalizer" for type "Tokenizer". Property "normalizer" has no defined setter
    tokenizer.normalizer = tokenizer_params.normalizer  # type: ignore
    tokenizer.pre_tokenizer = tokenizer_params.pre_tokenizer  # type: ignore
    tokenizer.decoder = tokenizer_params.decoder  # type: ignore
    tokenizer_trainer = tokenizer_params.trainer_cls(
        vocab_size=vocab_size,
        initial_alphabet=tokenizer_params.initial_alphabet,
        special_tokens=tokenizer_params.special_tokens,
    )

    tokenizer.train_from_iterator(flattened_dataset, trainer=tokenizer_trainer)

    # re-sort the tokenizer vocabulary in order to create determinism in the dataloader
    # TODO: handle the GPT case in the future
    if isinstance(tokenizer.model, tokenizers_models.WordPiece):
        vocab = tokenizer.get_vocab()
        # start by deleting the special tokens from the vocab map
        for token in tokenizer_trainer.special_tokens:
            del vocab[token.content]
        # re-assign token indicies
        for idx, vocab_item in enumerate(sorted(vocab.keys())):
            vocab[vocab_item] = idx + len(tokenizer_trainer.special_tokens)
        # add special tokens back in
        for idx, token in enumerate(tokenizer_trainer.special_tokens):
            vocab[token.content] = idx

        tokenizer.model = tokenizer.model.__class__(vocab)  # type: ignore

    # save the tokenizer config
    with TemporaryDirectory() as tmp_path:
        tmp_tokenizer_dir = str(tmp_path)
        tmp_tokenizer_file = join(tmp_tokenizer_dir, 'tokenizer.json')
        tokenizer.save(tmp_tokenizer_file)  #type: ignore (thirdparty)

        # save the vocabulary and potential merges file
        tokenizer.model.save(tmp_tokenizer_dir)  # type: ignore

        # the .from_pretrained method doesn't load our padding for some reason, so we save it as a special kwarg
        tmp_tokenizer_config = join(tmp_tokenizer_dir, 'tokenizer_config.json')
        with open(tmp_tokenizer_config, 'w') as f:
            json.dump({'pad_token': tokenizer_params.pad_token}, f)

        # instantiate the new tokenizer
        if not issubclass(tokenizer_params.tokenizer_cls, PreTrainedTokenizer):
            raise ValueError(f'{tokenizer_params.tokenizer_cls} should sub-class transformers.PreTrainedTokenizer.')
        # print("Temporary path:", tmp_path)
        # input("Waiting for input..")
        tokenizer = tokenizer_params.tokenizer_cls.from_pretrained(  #type: ignore
            tmp_tokenizer_dir)

    return tokenizer


def synthetic_hf_dataset_builder(num_samples: int, chars_per_sample: int, column_names: list, seed=5):
    """Creates a synthetic :class:`~datasets.Dataset` and passes it to the preprocessing scripts.

    Args:
        num_samples (int): how many samples to use in the synthetic dataset.
        chars_per_sample (int): how many characters each synthetic text sample should be.
        column_names (list): the column names that a dataset should use

    Returns:
        datasets.Dataset: the synthetic HF Dataset object.
    """

    try:
        import datasets
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

    if column_names is None or len(column_names) == 0:
        raise ValueError('There must be at least one column name provided for the final dataset.')

    data = {}
    random_generator = Random(seed)
    for column_name in column_names:
        data[column_name] = [
            _generate_synthetic_text_sample(chars_per_sample, random_generator) for _ in range(num_samples)
        ]
    data['idx'] = list(range(num_samples))

    hf_synthetic_dataset = datasets.Dataset.from_dict(data)
    return hf_synthetic_dataset


def _generate_synthetic_text_sample(chars_per_sample, random_generator, min_word_length=3, max_word_length=10):
    character_set = {
        'letters': {
            'weight': 10,
            'choices': string.ascii_letters
        },
        'digits': {
            'weight': 5,
            'choices': string.digits
        },
        'punctuation': {
            'weight': 1,
            'choices': string.punctuation
        }
    }
    valid_chars = ''.join([(i['choices'] * i['weight']) for i in character_set.values()])

    sample = ''
    while len(sample) < chars_per_sample:
        sample_len = random_generator.randint(min_word_length, max_word_length)
        sample += ''.join([random_generator.choice(valid_chars) for _ in range(sample_len)])
        sample += ' '
    sample = sample[:chars_per_sample]
    return sample
