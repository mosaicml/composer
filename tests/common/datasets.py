# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import List, Optional, Sequence

import pytest
import torch
import torch.utils.data
import yahp as hp
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

from composer.datasets import GLUEHparams, LMDatasetHparams
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.models import ModelHparams
from composer.models.transformer_hparams import TransformerHparams
from tests.common.models import model_hparams_to_tokenizer_family


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (5, 1, 1))
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


@dataclasses.dataclass
class RandomClassificationDatasetHparams(DatasetHparams, SyntheticHparamsMixin):

    data_shape: List[int] = hp.optional("data shape", default_factory=lambda: [1, 1, 1])
    num_classes: int = hp.optional("num_classes", default=2)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams):
        assert self.data_shape is not None
        assert self.num_classes is not None
        dataset = RandomClassificationDataset(
            size=self.synthetic_num_unique_samples,
            shape=self.data_shape,
            num_classes=self.num_classes,
        )
        if self.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        return dataloader_hparams.initialize_object(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=self.drop_last,
        )


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

        self.size = size
        self.x = torch.randn(size, *shape)
        self.y = torch.randint(0, num_classes, size=(size,))

        super().__init__(root='')

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        x = self.x[index]
        y = self.y[index]

        if self.is_PIL:
            x = x.numpy()
            x = (x - x.min())
            x = (x * (255 / x.max())).astype("uint8")
            x = Image.fromarray(x)

        if self.transform is not None:
            return self.transform(x), y
        else:
            return x, y


def configure_dataset_hparams_for_synthetic(
    dataset_hparams: DatasetHparams,
    model_hparams: Optional[ModelHparams] = None,
) -> None:
    if not isinstance(dataset_hparams, SyntheticHparamsMixin):
        pytest.xfail(f"{dataset_hparams.__class__.__name__} does not support synthetic data or num_total_batches")

    assert isinstance(dataset_hparams, SyntheticHparamsMixin)

    dataset_hparams.use_synthetic = True

    if isinstance(model_hparams, TransformerHparams):
        if type(model_hparams) not in model_hparams_to_tokenizer_family:
            raise ValueError(f"Model {type(model_hparams)} is currently not supported for synthetic testing!")

        tokenizer_family = model_hparams_to_tokenizer_family[type(model_hparams)]
        assert isinstance(dataset_hparams, (GLUEHparams, LMDatasetHparams))
        dataset_hparams.tokenizer_name = tokenizer_family
        dataset_hparams.max_seq_length = 128
