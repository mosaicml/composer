# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Generic hyperparameters for self-supervised training of autoregressive and masked language models."""

import logging
from dataclasses import asdict, dataclass
from typing import List, Optional

import yahp as hp
from torch.utils.data import DataLoader

from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from composer.datasets.lm_dataset import build_lm_dataloader, build_synthetic_lm_dataloader
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin

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
        max_seq_length: (int, optional): Custom sequence length for the training dataset.
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
                tokenizer_name=self.tokenizer_name,  # type: ignore
                batch_size=batch_size,
                split=self.split,  # type: ignore
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                use_masked_lm=self.use_masked_lm,
                num_tokens=self.num_tokens,
                mlm_probability=self.mlm_probability,
                subsample_ratio=self.subsample_ratio,
                max_seq_length=self.max_seq_length,
                **asdict(dataloader_hparams),
            )
        else:
            return build_lm_dataloader(
                datadir=self.datadir,
                tokenizer_name=self.tokenizer_name,  # type: ignore
                batch_size=batch_size,
                split=self.split,  # type: ignore
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                use_masked_lm=self.use_masked_lm,
                num_tokens=self.num_tokens,
                mlm_probability=self.mlm_probability,
                subsample_ratio=self.subsample_ratio,
                **asdict(dataloader_hparams),
            )
