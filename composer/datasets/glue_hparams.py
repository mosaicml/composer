# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""GLUE (General Language Understanding Evaluation) dataset hyperparameters (Wang et al, 2019).

The GLUE benchmark datasets consist of nine sentence- or sentence-pair language
understanding tasks designed to cover a diverse range of dataset sizes, text genres, and
degrees of difficulty.

Note that the GLUE diagnostic dataset, which is designed to evaluate and analyze model
performance with respect to a wide range of linguistic phenomena found in natural
language, is not included here.

Please refer to the `GLUE`_ benchmark for more details.

.. _GLUE: https://gluebenchmark.com/
"""

import logging
from dataclasses import dataclass
from typing import Optional, cast

import yahp as hp
from torch.utils.data import DataLoader

from composer.core.types import Dataset
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.datasets.synthetic_lm import generate_synthetic_tokenizer, synthetic_hf_dataset_builder
from composer.utils import MissingConditionalImportError, dist

__all__ = ['GLUEHparams']

log = logging.getLogger(__name__)

_task_to_keys = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
}


@dataclass
class GLUEHparams(DatasetHparams, SyntheticHparamsMixin):
    """Sets up a generic GLUE dataset loader.

    Args:
        task (str): the GLUE task to train on, choose one from: ``'CoLA'``, ``'MNLI'``,
            ``'MRPC'``, ``'QNLI'``, ``'QQP'``, ``'RTE'``, ``'SST-2'``, and ``'STS-B'``.
        tokenizer_name (str): The name of the HuggingFace tokenizer to preprocess text
            with. See `HuggingFace documentation <https://huggingface.co/models>`_.
        split (str): Whether to use ``'train'``, ``'validation'``, or ``'test'`` split.
        max_seq_length (int, optional): Optionally, the ability to set a custom sequence
            length for the training dataset. Default: ``256``.
        max_network_retries (int, optional): Number of times to retry HTTP requests if
            they fail. Default: ``10``.

    Returns:
        DataLoader: A PyTorch :class:`~torch.utils.data.DataLoader` object.
    """

    task: Optional[str] = hp.optional(
        'The GLUE task to train on, choose one from: CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, and STS-B.', default=None)
    tokenizer_name: Optional[str] = hp.optional('The name of the HuggingFace tokenizer to preprocess text with.',
                                                default=None)
    split: Optional[str] = hp.optional("Whether to use 'train', 'validation' or 'test' split.", default=None)
    max_seq_length: int = hp.optional(
        default=256, doc='Optionally, the ability to set a custom sequence length for the training dataset.')
    max_network_retries: int = hp.optional(default=10,
                                           doc='Optionally, the number of times to retry HTTP requests if they fail.')

    def validate(self):
        if self.task not in _task_to_keys.keys():
            raise ValueError(f"The task must be a valid GLUE task, options are {' ,'.join(_task_to_keys.keys())}.")

        if (self.max_seq_length % 8) != 0:
            log.warning('For best hardware acceleration, it is recommended that sequence lengths be multiples of 8.')

        if self.tokenizer_name is None:
            raise ValueError('A tokenizer name must be specified to tokenize the dataset.')

        if self.split is None:
            raise ValueError('A dataset split must be specified.')

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        # TODO (Moin): I think this code is copied verbatim in a few different places. Move this into a function.
        try:
            import datasets
            import transformers
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers') from e

        self.validate()
        assert self.task is not None, 'checked in validate()'
        assert self.tokenizer_name is not None, 'checked in validate()'

        if self.use_synthetic:
            column_names = [i for i in _task_to_keys[self.task] if i is not None]

            # we just use the max sequence length in tokens to upper bound the sequence length in characters
            dataset = synthetic_hf_dataset_builder(num_samples=self.synthetic_num_unique_samples,
                                                   chars_per_sample=self.max_seq_length,
                                                   column_names=column_names)

            # flatten the columnar dataset into one column
            tokenizer = generate_synthetic_tokenizer(tokenizer_family=self.tokenizer_name, dataset=dataset)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)  #type: ignore (thirdparty)

            log.info(f'Loading {self.task.upper()} on rank {dist.get_global_rank()}')
            download_config = datasets.utils.DownloadConfig(max_retries=self.max_network_retries)
            dataset = datasets.load_dataset('glue', self.task, split=self.split, download_config=download_config)

        log.info(f'Starting tokenization step by preprocessing over {dataloader_hparams.num_workers} threads!')
        text_column_names = _task_to_keys[self.task]

        def tokenize_function(inp):
            # truncates sentences to max_length or pads them to max_length

            first_half = inp[text_column_names[0]]
            second_half = inp[text_column_names[1]] if text_column_names[1] in inp else None
            return tokenizer(
                text=first_half,
                text_pair=second_half,
                padding='max_length',
                max_length=self.max_seq_length,
                truncation=True,
            )

        columns_to_remove = ['idx'] + [i for i in text_column_names if i is not None]

        assert isinstance(dataset, datasets.Dataset)
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=None if dataloader_hparams.num_workers == 0 else dataloader_hparams.num_workers,
            batch_size=1000,
            remove_columns=columns_to_remove,
            new_fingerprint=f'{self.task}-{self.tokenizer_name}-tokenization-{self.split}',
            load_from_cache_file=True,
        )

        data_collator = transformers.default_data_collator
        sampler = dist.get_sampler(cast(Dataset, dataset), drop_last=self.drop_last, shuffle=self.shuffle)

        return dataloader_hparams.initialize_object(
            dataset=dataset,  #type: ignore (thirdparty)
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
            collate_fn=data_collator)
