# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import DataLoader

from composer.utils import dist
from tests.common.datasets import RandomTextClassificationDataset, RandomTextLMDataset


def dummy_transformer_classifier_batch(vocab_size=100, num_classes=2):
    sequence_length = 32
    size = 100
    batch_size = 8
    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    return next(iter(train_dataloader))


def dummy_tiny_bert_classification_batch():
    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    num_classes = 2
    size = 16
    batch_size = 8

    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes,
                                                    use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    batch = next(iter(train_dataloader))
    return batch


def dummy_tiny_bert_lm_batch():
    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    size = 16
    batch_size = 8

    train_dataset = RandomTextLMDataset(size=size,
                                        vocab_size=vocab_size,
                                        sequence_length=sequence_length,
                                        use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    batch = next(iter(train_dataloader))
    return batch
