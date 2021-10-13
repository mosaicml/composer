# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.collections import MetricCollection

from composer.core.types import BatchPair
from composer.models.loss import CrossEntropyLoss, soft_cross_entropy

if TYPE_CHECKING:
    from composer.core.types import Batch, Metrics, Tensors


class BaseMosaicModel(torch.nn.Module, abc.ABC):

    @abc.abstractmethod
    def loss(self, outputs: Any, batch: Batch, *args, **kwargs) -> Tensors:
        """Compute the loss of the model.

        Args:
            outputs: output of the foward pass
            batch: input batch from dataloader

        Returns:
            The loss as a Tensors object.
        """
        pass

    @abc.abstractmethod
    def forward(self, batch: Batch) -> Tensors:
        """Compute model output given an input.

        Args:
            batch (Batch): input batch for forward pass

        Returns:
            Tensors: result that is passed to loss
        """
        pass

    @abc.abstractmethod
    def metrics(self, train: bool = False) -> Metrics:
        """Used to get metrics that will be used to evaluate the model. Note
        that each metric keeps states which are updated with data seen so far.
        As a result, different metric objects should be used for training and validation.
        See: https://torchmetrics.readthedocs.io/en/latest/pages/overview.html
        for more details.

        Args:
            train (optional): True to return metrics that should be computed during training
            and False otherwise. Default is False.

        Returns:
            Either a torchmetrics.Metric object or a torchmetrics.MetricCollection
            object.
        """
        pass

    @abc.abstractmethod
    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        """
        Compute model outputs given data. The output of this function
        will be used as input to any metrics being computed for the model.

        Args:
            batch: The data to perform validation with. Specified as a tuple of
            tensors (input, target).

        Returns:
            Tuple[Any, Any]: tuple that is passed directly
                             to the provided torchmetrics.Metric's update method.
        """
        pass


class MosaicClassifier(BaseMosaicModel):

    num_classes: Optional[int] = None

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_loss = CrossEntropyLoss()
        self.module = module

        # TODO(issue #249): Needs to be replaced
        if hasattr(self.module, "num_classes"):
            self.num_classes = getattr(self.module, "num_classes")

    def loss(self, outputs: Any, batch: BatchPair, *args, **kwargs) -> Tensors:
        _, y = batch
        assert isinstance(outputs, Tensor), "Loss expects outputs as Tensor"
        assert isinstance(y, Tensor), "Loss does not support multiple target Tensors"
        return soft_cross_entropy(outputs, y, *args, **kwargs)

    def metrics(self, train: bool = False) -> Metrics:
        return self.train_acc if train else MetricCollection([self.val_acc, self.val_loss])

    def forward(self, batch: BatchPair) -> Tensor:
        x, y = batch
        logits = self.module(x)

        return logits

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        assert self.training is False, "For validation, model must be in eval mode"
        inputs, targets = batch
        logits = self.forward(batch)
        return logits, targets
