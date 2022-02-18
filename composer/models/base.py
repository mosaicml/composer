# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.collections import MetricCollection

from composer.core.types import Batch, BatchPair, Metrics, Tensors
from composer.models.loss import CrossEntropyLoss, soft_cross_entropy

__all__ = ["ComposerModel", "ComposerClassifier"]


class ComposerModel(torch.nn.Module, abc.ABC):
    """The minimal interface needed to use a model with :class:`composer.trainer.Trainer`."""

    @abc.abstractmethod
    def loss(self, outputs: Any, batch: Batch, *args, **kwargs) -> Tensors:
        """Compute the loss of the model.

        Args:
            outputs (Any): The output of the forward pass.
            batch (~composer.core.types.Batch): The input batch from dataloader.

        Returns:
            Tensors:
                The loss as a ``Tensors`` object.
        """
        pass

    @abc.abstractmethod
    def forward(self, batch: Batch) -> Tensors:
        """Compute model output given an input.

        Args:
            batch (Batch): The input batch for the forward pass.

        Returns:
            Tensors:
                The result that is passed to :meth:`loss` as a ``Tensors``
                object.
        """
        pass

    def metrics(self, train: bool = False) -> Metrics:
        """Get metrics for evaluating the model.

        .. warning:: Each metric keeps states which are updated with data seen so far.
                     As a result, different metric instances should be used for training
                     and validation. See:
                     https://torchmetrics.readthedocs.io/en/latest/pages/overview.html
                     for more details.

        Args:
            train (bool, optional): True to return metrics that should be computed
                during training and False otherwise. (default: ``False``)

        Returns:
            Metrics: A ``Metrics`` object.
        """
        raise NotImplementedError('Implement metrics in your ComposerModel to run validation.')

    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        """Compute model outputs on provided data.

        The output of this function will be directly used as input
        to all metrics returned by :meth:`metrics`.

        Args:
            batch (Batch): The data to perform validation with.
                Specified as a tuple of tensors (input, target).

        Returns:
            Tuple[Any, Any]: Tuple that is passed directly to the
                `update()` methods of the metrics returned by :meth:`metrics`.
                Most often, this will be a tuple of the form (predictions, targets).
        """
        raise NotImplementedError('Implement validate in your ComposerModel to run validation.')


class ComposerClassifier(ComposerModel):
    """Implements the base logic that all classifiers can build on top of.

    Inherits from :class:`~composer.models.ComposerModel`.

    Args:
        module (torch.nn.Module): The neural network module to wrap with
            :class:`~composer.models.ComposerClassifier`.
    """

    num_classes: Optional[int] = None

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_loss = CrossEntropyLoss()
        self.module = module

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
        x, _ = batch
        logits = self.module(x)

        return logits

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        assert self.training is False, "For validation, model must be in eval mode"
        _, targets = batch
        logits = self.forward(batch)
        return logits, targets
