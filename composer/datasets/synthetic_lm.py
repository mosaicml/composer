# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import json
import random
import string
import textwrap
from os.path import join
from tempfile import mkdtemp
from typing import TYPE_CHECKING, NamedTuple, Optional

if TYPE_CHECKING:
    import tokenizers.decoders as decoders
    import tokenizers.models as tokenizers_models
    import tokenizers.normalizers as normalizers
    import tokenizers.pre_tokenizers as pre_tokenizers
    from datasets import Dataset
    from transformers import PreTrainedTokenizer


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
        import tokenizers.decoders as decoders
        import tokenizers.models as tokenizers_models
        import tokenizers.normalizers as normalizers
        import tokenizers.pre_tokenizers as pre_tokenizers
        import tokenizers.trainers as tokenizers_trainer
        from transformers import BertTokenizer
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
            if using pip or `conda install -c conda-forge transformers tokenizers` if using Anaconda.""")) from e

    unk_token = "[UNK]"
    pad_token = "[PAD]"

    initial_alphabet = "".join([i for i in dataset])
    initial_alphabet = list(set(initial_alphabet))

    return SyntheticTokenizerParams(
        tokenizer_model=tokenizers_models.WordPiece(unk_token=unk_token),  # type: ignore
        normalizer=normalizers.BertNormalizer(),
        pre_tokenizer=pre_tokenizers.BertPreTokenizer(),
        decoder=decoders.WordPiece(),
        initial_alphabet=initial_alphabet,
        special_tokens=[pad_token, unk_token, "[SEP]", "[CLS]", "[MASK]"],
        pad_token=pad_token,
        trainer_cls=tokenizers_trainer.WordPieceTrainer,
        tokenizer_cls=BertTokenizer,
    )


def _generate_gpt2_tokenizer_params() -> SyntheticTokenizerParams:
    try:
        import tokenizers.decoders as decoders
        import tokenizers.models as tokenizers_models
        import tokenizers.normalizers as normalizers
        import tokenizers.pre_tokenizers as pre_tokenizers
        import tokenizers.trainers as tokenizers_trainer
        from transformers import GPT2Tokenizer
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
            if using pip or `conda install -c conda-forge tokenizers` if using Anaconda.""")) from e

    unk_token = None
    pad_token = "<pad>"

    return SyntheticTokenizerParams(
        tokenizer_model=tokenizers_models.BPE(unk_token=unk_token),
        normalizer=normalizers.Lowercase(),
        pre_tokenizer=pre_tokenizers.ByteLevel(),
        decoder=decoders.ByteLevel(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[pad_token, "<endoftext>"],
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
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizer
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
            if using pip or `conda install -c conda-forge transformers tokenizers` if using Anaconda.""")) from e

    # generate a synthetic dataset with reasonable defaults is none is provided
    if dataset is None:
        num_samples = 100
        chars_per_sample = 128
        column_names = ['sentence']
        dataset = SyntheticHFDataset(num_samples=num_samples,
                                     chars_per_sample=chars_per_sample,
                                     column_names=column_names).generate_dataset()

    # change a columnar dataset into a list
    flattened_columns = [dataset[key] for key in dataset.column_names if key != 'idx']
    # flatten the list of lists into a single list
    flattened_dataset = []
    for sublist in flattened_columns:
        for item in sublist:
            flattened_dataset.append(item)

    if "bert" in tokenizer_family:
        tokenizer_params = _generate_bert_tokenizer_params(flattened_dataset)
    elif "gpt2" in tokenizer_family:
        tokenizer_params = _generate_gpt2_tokenizer_params()
    else:
        raise ValueError(f"Synthetic tokenizers for tokenizer family {tokenizer_family} are currently unsupported.")

    tokenizer = Tokenizer(tokenizer_params.tokenizer_model)
    tokenizer.enable_padding(direction="right",
                             pad_id=0,
                             pad_type_id=0,
                             pad_token=tokenizer_params.pad_token,
                             pad_to_multiple_of=8)
    tokenizer.normalizer = tokenizer_params.normalizer  # type: ignore
    tokenizer.pre_tokenizer = tokenizer_params.pre_tokenizer  # type: ignore
    tokenizer.decoder = tokenizer_params.decoder  # type: ignore
    tokenizer_trainer = tokenizer_params.trainer_cls(
        vocab_size=vocab_size,
        initial_alphabet=tokenizer_params.initial_alphabet,
        special_tokens=tokenizer_params.special_tokens,
    )
    tokenizer.train_from_iterator(flattened_dataset, trainer=tokenizer_trainer)

    # save the tokenizer config
    tmp_tokenizer_dir = mkdtemp()
    tmp_tokenizer_file = join(tmp_tokenizer_dir, "tokenizer.json")
    tokenizer.save(tmp_tokenizer_file)

    # save the vocabulary and potential merges file
    tokenizer_params.tokenizer_model.save(tmp_tokenizer_dir)  # type: ignore

    # the .from_pretrained method doesn't load our padding for some reason, so we save it as a special kwarg
    tmp_tokenizer_config = join(tmp_tokenizer_dir, "tokenizer_config.json")
    with open(tmp_tokenizer_config, "w") as f:
        json.dump({"pad_token": tokenizer_params.pad_token}, f)

    # instantiate the new tokenizer
    if not issubclass(tokenizer_params.tokenizer_cls, PreTrainedTokenizer):
        raise ValueError(f"{tokenizer_params.tokenizer_cls} should sub-class transformers.PreTrainedTokenizer.")
    tokenizer = tokenizer_params.tokenizer_cls.from_pretrained(tmp_tokenizer_dir)

    return tokenizer


class SyntheticHFDataset:
    """Creates a synthetic HF dataset and passes it to the preprocessing scripts."""

    def __init__(self, num_samples: int, chars_per_sample: int, column_names: list):
        if column_names is None or len(column_names) == 0:
            raise ValueError("There must be at least one column name provided for the final dataset.")
        self.num_samples = num_samples
        self.chars_per_sample = chars_per_sample
        self.column_names = column_names

    def generate_dataset(self):
        try:
            import datasets
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without NLP support. To use NLP with Composer, run `pip install mosaicml[nlp]`
                if using pip or `conda install -c conda-forge transformers` if using Anaconda.""")) from e

        data = {}
        for column_name in self.column_names:
            data[column_name] = [self.generate_sample() for _ in range(self.num_samples)]
        data['idx'] = list(range(self.num_samples))

        hf_synthetic_dataset = datasets.Dataset.from_dict(data)
        return hf_synthetic_dataset

    def generate_sample(self):
        MIN_WORD_LENGTH = 3
        MAX_WORD_LENGTH = 10
        character_set = {
            "letters": {
                "weight": 10,
                "choices": string.ascii_letters
            },
            "digits": {
                "weight": 5,
                "choices": string.digits
            },
            "punctuation": {
                "weight": 1,
                "choices": string.punctuation
            }
        }
        valid_chars = ''.join([(i['choices'] * i['weight']) for i in character_set.values()])

        sample = ''
        while len(sample) < self.chars_per_sample:
            sample_len = random.randint(MIN_WORD_LENGTH, MAX_WORD_LENGTH)
            sample += ''.join([random.choice(valid_chars) for _ in range(sample_len)])
            sample += ' '
        sample = sample[:self.chars_per_sample]
        return sample
