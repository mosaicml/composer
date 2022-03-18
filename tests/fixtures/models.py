# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
import yahp as hp
from torchmetrics import Metric, MetricCollection

from composer.core.types import BatchPair, DataLoader
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic import SyntheticBatchPairDataset, SyntheticDataLabelType, SyntheticPILDataset
from composer.models import ComposerModel, ModelHparams


class SimpleBatchPairModel(ComposerModel):
    """A small model that has a really fast forward pass."""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = torch.nn.Linear(num_channels, 5)

        self.fc2 = torch.nn.Linear(5, num_classes)

        self.net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            self.fc1,
            torch.nn.ReLU(),
            self.fc2,
            torch.nn.Softmax(dim=-1),
        )

    def loss(self, outputs: torch.Tensor, batch: BatchPair, *args,
             **kwargs) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        _, target = batch
        assert isinstance(target, torch.Tensor)
        return F.cross_entropy(outputs, target, *args, **kwargs)

    def validate(self, batch: BatchPair) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = batch
        assert isinstance(x, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        pred = self.forward(batch)
        return pred, target

    def forward(self, batch: BatchPair) -> torch.Tensor:
        x, _ = batch
        return self.net(x)

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        if train:
            return self.train_acc
        else:
            return self.val_acc


@dataclasses.dataclass
class _SimpleDatasetHparams(DatasetHparams, SyntheticHparamsMixin):

    data_shape: Optional[List[int]] = hp.optional("data shape", default=None)
    num_classes: Optional[int] = hp.optional("num_classes", default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        assert self.data_shape is not None
        assert self.num_classes is not None
        dataset = SyntheticBatchPairDataset(total_dataset_size=10_000,
                                            data_shape=self.data_shape,
                                            label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                                            num_classes=self.num_classes,
                                            memory_format=self.synthetic_memory_format,
                                            num_unique_samples_to_create=self.synthetic_num_unique_samples,
                                            device=self.synthetic_device)
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


@dataclasses.dataclass
class _SimplePILDatasetHparams(DatasetHparams, SyntheticHparamsMixin):

    data_shape: Optional[List[int]] = hp.optional("data shape", default=None)
    num_classes: Optional[int] = hp.optional("num_classes", default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataLoaderHparams) -> DataLoader:
        assert self.data_shape is not None
        assert self.num_classes is not None
        dataset = SyntheticPILDataset(total_dataset_size=10_000,
                                      data_shape=tuple(self.data_shape),
                                      label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
                                      num_classes=self.num_classes,
                                      num_unique_samples_to_create=self.synthetic_num_unique_samples)
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


@dataclass
class _SimpleBatchPairModelHparams(ModelHparams):
    num_channels: int = hp.optional("number of image channels", default_factory=lambda: 3)
    num_classes: int = hp.optional("number of output classes", default=10)

    def initialize_object(self) -> SimpleBatchPairModel:
        return SimpleBatchPairModel(
            num_channels=self.num_channels,
            num_classes=self.num_classes,
        )


class SimpleConvModel(torch.nn.Module):
    """Very basic forward operation with no activation functions Used just to test that model surgery doesn't create
    forward prop bugs."""

    def __init__(self):
        super().__init__()

        conv_args = dict(kernel_size=(3, 3), padding=1)
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=8, stride=2, bias=False, **conv_args)  # stride > 1
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, stride=2, bias=False,
                                     **conv_args)  # stride > 1 but in_channels < 16
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, bias=False, **conv_args)  # stride = 1

        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        self.pool2 = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 48)
        self.linear2 = torch.nn.Linear(48, 10)

    def forward(self, x: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Union[torch.Tensor, Sequence[torch.Tensor]]:

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
