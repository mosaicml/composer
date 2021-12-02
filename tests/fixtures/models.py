# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.utils.data
import torchmetrics
import yahp as hp

from composer.core.types import BatchPair, DataLoader, Metrics, Tensor, Tensors
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams, SyntheticHparamsMixin
from composer.datasets.synthetic import SyntheticBatchPairDataset, SyntheticDataLabelType
from composer.models import BaseMosaicModel, ModelHparams


class SimpleBatchPairModel(BaseMosaicModel):
    """A small model that has a really fast forward pass.
    """

    def __init__(self, in_shape: Tuple[int, ...], num_classes: int) -> None:
        super().__init__()

        self.in_shape = in_shape
        in_features_flattened = 1
        for dim in self.in_shape:
            in_features_flattened *= dim
        self.num_classes = num_classes

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features_flattened, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, num_classes),
            torch.nn.Softmax(dim=-1),
        )

    def loss(self, outputs: Tensor, batch: BatchPair) -> Tensors:
        _, target = batch
        return self.loss_fn(outputs, target)

    def validate(self, batch: BatchPair) -> Tuple[Tensor, Tensor]:
        x, target = batch
        assert isinstance(x, Tensor)
        assert isinstance(target, Tensor)
        pred = self.forward(batch)
        return pred, target

    def forward(self, batch: BatchPair) -> Tensor:
        x, _ = batch
        return self.net(x)

    def metrics(self, train: bool = False) -> Metrics:
        if train:
            return self.train_acc
        else:
            return self.val_acc


@dataclasses.dataclass
class _SimpleDatasetHparams(DatasetHparams, SyntheticHparamsMixin):

    data_shape: Optional[List[int]] = hp.optional("data shape", default=None)
    num_classes: Optional[int] = hp.optional("num_classes", default=None)

    def initialize_object(self, batch_size: int, dataloader_hparams: DataloaderHparams) -> DataLoader:
        assert self.subset_num_batches is not None
        assert self.data_shape is not None
        assert self.num_classes is not None
        total_dataset_size = self.subset_num_batches * batch_size
        dataset = SyntheticBatchPairDataset(total_dataset_size=total_dataset_size,
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


@dataclass
class _SimpleBatchPairModelHparams(ModelHparams):
    in_shape: List[int] = hp.optional("shape for a single input", default_factory=lambda: [10])
    num_classes: int = hp.optional("number of output classes", default=10)

    def initialize_object(self) -> SimpleBatchPairModel:
        return SimpleBatchPairModel(
            in_shape=tuple(self.in_shape),
            num_classes=self.num_classes,
        )


class SimpleConvModel(torch.nn.Module):
    """Very basic forward operation with no activation functions
    Used just to test that model surgery doesn't create forward prop bugs.
    """

    def __init__(self):
        super().__init__()

        conv_args = dict(kernel_size=(3, 3), padding=1)
        self.conv1 = torch.nn.Conv2d(in_channels=32, out_channels=8, stride=2, bias=False, **conv_args)  # stride > 1
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=32, stride=2, bias=False,
                                     **conv_args)  # stride > 1 but in_channels < 16
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, stride=1, bias=False, **conv_args)  # stride = 1

        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)

    def forward(self, x: Tensors) -> Tensors:

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        return out
