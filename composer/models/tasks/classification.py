# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.

:class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.
"""

import logging
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy

from composer.loss import soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerModel

__all__ = ['ComposerClassifier']

log = logging.getLogger(__name__)


class ComposerClassifier(ComposerModel):
    """A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.
    :class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
    classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.

    Args:
        module (torch.nn.Module): A PyTorch neural network module.
        loss_name (str, optional): Loss function to use. E.g. 'soft_cross_entropy' or
            'binary_cross_entropy_with_logits'. Loss function must be in
            :mod:`~composer.loss.loss`. Default: ``'soft_cross_entropy'``".

    Returns:
        ComposerClassifier: An instance of :class:`.ComposerClassifier`.

    Example:

    .. testcode::

        import torchvision
        from composer.models import ComposerClassifier

        pytorch_model = torchvision.models.resnet18(pretrained=False)
        model = ComposerClassifier(pytorch_model)
    """

    num_classes: Optional[int] = None

    def __init__(self,
                 module: torch.nn.Module,
                 train_metrics: Optional[Union[Metric, MetricCollection]] = None,
                 val_metrics: Optional[Union[Metric, MetricCollection]] = None,
                 loss_fn: Callable = soft_cross_entropy) -> None:
        super().__init__()

        # Metrics for training
        if train_metrics is None:
            train_metrics = Accuracy()
        self.train_metrics = train_metrics

        # Metrics for validation
        if val_metrics is None:
            val_metrics = MetricCollection([CrossEntropy(), Accuracy()])
        self.val_metrics = val_metrics

        self.module = module
        self._loss_fn = loss_fn

        if hasattr(self.module, 'num_classes'):
            self.num_classes = getattr(self.module, 'num_classes')

    def loss(self, outputs: Tensor, batch: Tuple[Any, Tensor], *args, **kwargs) -> Tensor:
        _, targets = batch
        return self._loss_fn(outputs, targets, *args, **kwargs)

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        return self.train_metrics if train else self.val_metrics

    def forward(self, batch: Tuple[Tensor, Any]) -> Tensor:
        inputs, _ = batch
        outputs = self.module(inputs)
        return outputs

    def validate(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        _, targets = batch
        outputs = self.forward(batch)
        return outputs, targets
