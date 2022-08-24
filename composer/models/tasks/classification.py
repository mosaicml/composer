# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.

:class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

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
    classification training loop with a loss function `loss_fn` which takes in the model's outputs and the labels.

    Args:
        module (torch.nn.Module): A PyTorch neural network module.
        train_metrics (Metric | MetricCollection, optional): A torchmetric or collection of torchmetrics to be
            computed on the training set throughout training.
        val_metrics (Metric | MetricCollection, optional): A torchmetric or collection of torchmetrics to be
            computed on the validation set throughout training.
        loss_fn (Callable, optional): Loss function to use. This loss function should have at least two arguments:
            1) the output of the model and 2) ``target`` i.e. labels from the dataset.

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

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        if isinstance(metrics, Metric):
            metrics_dict = {metrics.__class__.__name__: metrics}
        else:
            metrics_dict = {}
            for name, metric in metrics.items():
                assert isinstance(metric, Metric)
                metrics_dict[name] = metric

        return metrics_dict

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        _, targets = batch
        metric.update(outputs, targets)

    def forward(self, batch: Tuple[Tensor, Any]) -> Tensor:
        inputs, _ = batch
        outputs = self.module(inputs)
        return outputs

    def eval_forward(self, batch: Any, outputs: Optional[Any] = None) -> Any:
        return outputs if outputs is not None else self.forward(batch)
