# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from typing import Sequence

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision.datasets import VisionDataset

from composer.utils import dist
from tests.common.models import configure_tiny_bert_tokenizer, configure_tiny_gpt2_tokenizer


class InfiniteClassificationDataset(IterableDataset):
    """Classification dataset that never ends.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), num_classes: int = 2):
        self.shape = shape
        self.num_classes = num_classes

    def __iter__(self):
        while True:
            yield torch.randn(*self.shape), torch.randint(0, self.num_classes, size=(1,))[0]


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class RandomImageDataset(VisionDataset):
    """ Image Classification dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (32, 32, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (3, 32, 32), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        if is_PIL:  # PIL expects HWC
            shape = (shape[1], shape[2], shape[0])
        self.shape = shape
        self.num_classes = num_classes

        self.size = size
        self.x = None
        self.y = None

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype('uint8')
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y


class RandomSegmentationDataset(VisionDataset):
    """ Image Segmentation dataset with values drawn from a normal distribution
    Args:
        shape (Sequence[int]): shape of features. Defaults to (32, 32, 3)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        is_PIL (bool): if true, will emit image in PIL format (default: False)
    """

    def __init__(self, shape: Sequence[int] = (3, 32, 32), size: int = 100, num_classes: int = 2, is_PIL: bool = False):
        self.is_PIL = is_PIL
        if is_PIL:  # PIL expects HWC
            shape = (shape[1], shape[2], shape[0])
        self.shape = shape
        self.num_classes = num_classes

        self.size = size
        self.x = None
        self.y = None

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            mask_shape = self.shape[:2] if self.is_PIL else self.shape[1:]
            self.y = torch.randint(0, self.num_classes, size=(self.size, *mask_shape))
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype('uint8')
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y


class RandomTextClassificationDataset(Dataset):
    """ Text classification dataset with values (just input token ids) drawn uniformly
    Args:
        vocab_size (int): vocab size to use (default: 10)
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
        sequence_length (int): sequence length to use, all sequences will be of this length with no padding (default: 8)
        use_keys: (bool): whether to return the item in a dictionary with keys for input and output
    """

    def __init__(self,
                 size: int = 100,
                 vocab_size: int = 10,
                 sequence_length: int = 8,
                 num_classes: int = 2,
                 use_keys: bool = False):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.use_keys = use_keys

        self.input_key = 'input_ids'
        self.label_key = 'labels'

        self.size = size
        self.x = None
        self.y = None

        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if self.x is None:
            self.x = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))
        if self.y is None:
            self.y = torch.randint(low=0, high=self.num_classes, size=(self.size,))

        x = self.x[index]
        y = self.y[index]

        if self.use_keys:
            return {'input_ids': x, 'labels': y}
        else:
            return x, y


class RandomTextLMDataset(Dataset):
    """ Text LM dataset with values (just input token ids) drawn uniformly
    Args:
        vocab_size (int): vocab size to use (default: 10)
        size (int): number of samples (default: 100)
        sequence_length (int): sequence length to use, all sequences will be of this length with no padding (default: 8)
        use_keys: (bool): whether to return the item in a dictionary with keys for input and output
    """

    def __init__(self,
                 size: int = 100,
                 vocab_size: int = 10,
                 sequence_length: int = 8,
                 use_keys: bool = False,
                 use_token_type_ids: bool = True,
                 conditional_generation: bool = False):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.use_keys = use_keys
        self.use_token_type_ids = use_token_type_ids
        self.conditional_generation = conditional_generation

        self.input_key = 'input_ids'

        self.size = size
        self.x = None
        self.y = None

        super().__init__()

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        if self.x is None:
            self.x = torch.randint(low=0, high=self.vocab_size, size=(self.size, self.sequence_length))
            if self.conditional_generation:
                self.y = torch.randint(low=0, high=self.vocab_size, size=(self.size, 2 * self.sequence_length))

        x = self.x[index]

        if self.use_keys:
            output = {'input_ids': x}
            if self.use_token_type_ids:
                output['token_type_ids'] = torch.zeros_like(x)
            if self.y is not None:
                output['labels'] = self.y[index]
            return output
        else:
            return x if self.y is None else (x, self.y[index])


class SimpleDataset(Dataset):

    def __init__(self, size: int = 256, batch_size: int = 256, feature_size: int = 1, num_classes: int = 2):
        self.size = size
        self.batch_size = batch_size
        self.x = torch.randn(size * batch_size, feature_size)
        self.y = torch.randint(0, num_classes, size=(size * batch_size,), dtype=torch.long)

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        return self.x[index * self.batch_size:(index + 1) *
                      self.batch_size], self.y[index * self.batch_size:(index + 1) * self.batch_size]


def dummy_transformer_classifier_batch(vocab_size=100, num_classes=2):
    sequence_length = 32
    size = 8
    batch_size = 8
    train_dataset = RandomTextClassificationDataset(size=size,
                                                    vocab_size=vocab_size,
                                                    sequence_length=sequence_length,
                                                    num_classes=num_classes)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    return next(iter(train_dataloader))


def dummy_tiny_bert_classification_batch(num_classes=2):
    vocab_size = 30522  # Match bert vocab size
    sequence_length = 4
    size = 8
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
    size = 8
    batch_size = 8

    train_dataset = RandomTextLMDataset(size=size,
                                        vocab_size=vocab_size,
                                        sequence_length=sequence_length,
                                        use_keys=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=dist.get_sampler(train_dataset))
    batch = next(iter(train_dataloader))
    return batch


def dummy_hf_lm_dataloader(size: int, vocab_size: int, sequence_length: int, collate_fn=None):
    batch_size = 2

    dataset = RandomTextLMDataset(size=size, vocab_size=vocab_size, sequence_length=sequence_length, use_keys=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=dist.get_sampler(dataset), collate_fn=collate_fn)
    return dataloader


def dummy_bert_lm_dataloader(sequence_length=4, size=4):
    transformers = pytest.importorskip('transformers')
    tokenizer = configure_tiny_bert_tokenizer()
    collate_fn = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                                                 mlm=True,
                                                                                 mlm_probability=0.15)
    return dummy_hf_lm_dataloader(vocab_size=30522, sequence_length=sequence_length, size=size, collate_fn=collate_fn)


def dummy_gpt_lm_dataloader(sequence_length=4, size=4):
    transformers = pytest.importorskip('transformers')
    tokenizer = configure_tiny_gpt2_tokenizer()
    collate_fn = transformers.data.data_collator.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return dummy_hf_lm_dataloader(vocab_size=50257, sequence_length=sequence_length, size=size, collate_fn=collate_fn)


def dummy_text_classification_dataloader():
    dataset = RandomTextClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=4, sampler=dist.get_sampler(dataset))
    return dataloader
