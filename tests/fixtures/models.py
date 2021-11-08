# Copyright 2021 MosaicML. All Rights Reserved.

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.utils.data
import torchmetrics
import yahp as hp

from composer.core.types import BatchPair, Metrics, Tensor, Tensors
from composer.datasets.synthetic import SyntheticDataLabelType, SyntheticDatasetHparams
from composer.models import BaseMosaicModel, ModelHparams
from composer.trainer.trainer_hparams import TrainerHparams


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

    def get_dataset_hparams(self, total_dataset_size: int, drop_last: bool, shuffle: bool) -> SyntheticDatasetHparams:
        return SyntheticDatasetHparams(
            total_dataset_size=total_dataset_size,
            data_shape=list(self.in_shape),
            label_type=SyntheticDataLabelType.CLASSIFICATION_INT,
            num_classes=self.num_classes,
            device="cpu",
            drop_last=drop_last,
            shuffle=shuffle,
        )


@dataclass
class SimpleBatchPairModelHparams(ModelHparams):
    in_shape: List[int] = hp.optional("shape for a single input", default_factory=lambda: [10])
    num_classes: int = hp.optional("number of output classes", default=3)

    def initialize_object(self) -> SimpleBatchPairModel:
        return SimpleBatchPairModel(
            in_shape=tuple(self.in_shape),
            num_classes=self.num_classes,
        )


TrainerHparams.register_class("model", SimpleBatchPairModelHparams, "simple_batch_pair_model")


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
        self.pool2 = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(64, 48)
        self.linear2 = torch.nn.Linear(48, 10)

    def forward(self, x: Tensors) -> Tensors:

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        return out
