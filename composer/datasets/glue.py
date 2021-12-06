import logging
from dataclasses import dataclass
from multiprocessing import cpu_count

import numpy as np
import transformers
import yahp as hp
from tqdm import tqdm

from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.datasets.lm_datasets import _split_dict_fn

log = logging.getLogger(__name__)

@dataclass
class RTEHparams(DatasetHparams):
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    max_seq_length: int = hp.optional(
        default=256, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("Split must be one of 'train', 'validation', 'test'.")

        if (self.max_seq_length % 8) != 0:
            raise ValueError("For best performance, please ensure that sequence lengths are a multiple of eight.")

    def initialize_object(self) -> DataloaderSpec:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.dataset = datasets.load_dataset("glue", "rte", split=self.split)
        print("Loading RTE...")

        n_cpus = cpu_count()
        log.info(f"Starting tokenization step by preprocessing over {n_cpus} threads!")
        text_column_names = ["sentence1", "sentence2"]

        def tokenize_function(inp):
            # truncates sentences to max_length or pads them to max_length
            return self.tokenizer(
                text=inp[text_column_names[0]],
                text_pair=inp[text_column_names[1]],
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
            )

        columns_to_remove = ["idx"] + text_column_names
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_cpus,
            batch_size=1000,
            remove_columns=columns_to_remove,
            new_fingerprint=f"rte-tokenization-{self.split}",
            load_from_cache_file=True,
        )

        self.data_collator = transformers.data.data_collator.default_data_collator

        return DataloaderSpec(
            dataset=self.dataset,
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn,
        )


@dataclass
class QNLIHparams(DatasetHparams):
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    max_seq_length: int = hp.optional(
        default=256, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("Split must be one of 'train', 'validation', 'test'.")

        if (self.max_seq_length % 8) != 0:
            raise ValueError("For best performance, please ensure that sequence lengths are a multiple of eight.")

    def initialize_object(self) -> DataloaderSpec:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.dataset = datasets.load_dataset("glue", "qnli", split=self.split)
        print("Loading QNLI...")

        n_cpus = cpu_count()
        log.info(f"Starting tokenization step by preprocessing over {n_cpus} threads!")
        text_column_names = ["question", "sentence"]

        def tokenize_function(inp):
            # truncates sentences to max_length or pads them to max_length
            return self.tokenizer(
                text=inp[text_column_names[0]],
                text_pair=inp[text_column_names[1]],
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
            )

        columns_to_remove = ["idx"] + text_column_names
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_cpus,
            batch_size=1000,
            remove_columns=columns_to_remove,
            new_fingerprint=f"qnli-tokenization-{self.split}",
            load_from_cache_file=True,
        )

        self.data_collator = transformers.data.data_collator.default_data_collator

        return DataloaderSpec(
            dataset=self.dataset,
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn,
        )


@dataclass
class CoLAHparams(DatasetHparams):
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    max_seq_length: int = hp.optional(
        default=128, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("Split must be one of 'train', 'validation', 'test'.")

        if (self.max_seq_length % 8) != 0:
            raise ValueError("For best performance, please ensure that sequence lengths are a multiple of eight.")

    def initialize_object(self) -> DataloaderSpec:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.dataset = datasets.load_dataset("glue", "cola", split=self.split)
        print("Loading CoLA...")

        n_cpus = cpu_count()
        log.info(f"Starting tokenization step by preprocessing over {n_cpus} threads!")
        text_column_name = "sentence"

        def tokenize_function(inp):
            return self.tokenizer(
                inp[text_column_name],
                padding="max_length",
                max_length=self.max_seq_length,
            )

        columns_to_remove = ["idx", text_column_name]
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_cpus,
            batch_size=1000,
            remove_columns=columns_to_remove,
            new_fingerprint=f"cola-tokenization-{self.split}",
            load_from_cache_file=True,
        )

        self.data_collator = transformers.data.data_collator.default_data_collator

        return DataloaderSpec(
            dataset=self.dataset,
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn,
        )


@dataclass
class SST2Hparams(DatasetHparams):
    tokenizer_name: str = hp.required("The name of the tokenizer to preprocess text with.")
    split: str = hp.required("Whether to use 'train', 'validation' or 'test' split.")
    max_seq_length: int = hp.optional(
        default=128, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    shuffle: bool = hp.optional("Whether to shuffle the dataset for each epoch.", default=True)
    drop_last: bool = hp.optional("Whether to drop the last samples for the last batch.", default=False)

    def validate(self):
        if self.split not in ['train', 'validation', 'test']:
            raise ValueError("Split must be one of 'train', 'validation', 'test'.")

        if (self.max_seq_length % 8) != 0:
            raise ValueError("For best performance, please ensure that sequence lengths are a multiple of eight.")

    def initialize_object(self) -> DataloaderSpec:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)
        self.dataset = datasets.load_dataset("glue", "sst2", split=self.split)

        n_cpus = cpu_count()
        log.info(f"Starting tokenization step by preprocessing over {n_cpus} threads!")
        text_column_name = "sentence"

        def tokenize_function(inp):
            return self.tokenizer(
                inp[text_column_name],
                padding="max_length",
                max_length=self.max_seq_length,
            )

        columns_to_remove = ["idx", text_column_name]
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            num_proc=n_cpus,
            batch_size=1000,
            remove_columns=columns_to_remove,
            new_fingerprint=f"sst2-tokenization-{self.split}",
            load_from_cache_file=True,
        )

        self.data_collator = transformers.data.data_collator.default_data_collator

        return DataloaderSpec(
            dataset=self.dataset,
            collate_fn=self.data_collator,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            split_fn=_split_dict_fn,
        )
