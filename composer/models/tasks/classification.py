# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.

:class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.
"""

import logging
import textwrap
import warnings
from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy

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
        num_classes (int, optional): The number of output classes. Required if self.module does not have a num_classes parameter.
        train_metrics (Metric | MetricCollection, optional): A torchmetric or collection of torchmetrics to be
            computed on the training set throughout training. (default: :class:`MulticlassAccuracy`)
        val_metrics (Metric | MetricCollection, optional): A torchmetric or collection of torchmetrics to be
            computed on the validation set throughout training.
            (default: :class:`composer.metrics.CrossEntropy`, :class:`.MulticlassAccuracy`)
        loss_fn (Callable, optional): Loss function to use. This loss function should have at least two arguments:
            1) the output of the model and 2) ``target`` i.e. labels from the dataset.
    Returns:
        ComposerClassifier: An instance of :class:`.ComposerClassifier`.

    Example:

    .. testcode::

        import torchvision
        from composer.models import ComposerClassifier

        pytorch_model = torchvision.models.resnet18(pretrained=False)
        model = ComposerClassifier(pytorch_model, num_classes=1000)
    """

    num_classes: Optional[int] = None

    def __init__(
        self,
        module: torch.nn.Module,
        num_classes: Optional[int] = None,
        train_metrics: Optional[Union[Metric, MetricCollection]] = None,
        val_metrics: Optional[Union[Metric, MetricCollection]] = None,
        loss_fn: Callable = soft_cross_entropy,
    ) -> None:
        super().__init__()

        self.module = module
        self._loss_fn = loss_fn

        self.num_classes = num_classes
        if hasattr(self.module, 'num_classes'):
            model_num_classes = getattr(self.module, 'num_classes')
            if self.num_classes is not None and self.num_classes != model_num_classes:
                warnings.warn(
                    textwrap.dedent(
                        f'Specified num_classes={self.num_classes} does not match model num_classes={model_num_classes}.'
                        'Using model num_classes.',
                    ),
                )
            self.num_classes = model_num_classes
        if self.num_classes is None and (train_metrics is None or val_metrics is None):
            raise ValueError(
                textwrap.dedent(
                    'Please specify the number of output classes. Either: \n (1) pass '
                    'in num_classes to the ComposerClassifier \n (2) pass in both '
                    'train_metrics and val_metrics to Composer Classifier, or \n (3) '
                    'specify a num_classes parameter in the PyTorch network module.',
                ),
            )

        # Metrics for training
        if train_metrics is None:
            assert self.num_classes is not None
            train_metrics = MulticlassAccuracy(num_classes=self.num_classes, average='micro')
        self.train_metrics = train_metrics

        # Metrics for validation
        if val_metrics is None:
            assert self.num_classes is not None
            val_metrics = MetricCollection([
                CrossEntropy(),
                MulticlassAccuracy(num_classes=self.num_classes, average='micro'),
            ])
        self.val_metrics = val_metrics

    def loss(self, outputs: Tensor, batch: tuple[Any, Tensor], *args, **kwargs) -> Tensor:
        _, targets = batch
        return self._loss_fn(outputs, targets, *args, **kwargs)

    def get_metrics(self, is_train: bool = False) -> dict[str, Metric]:
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

    def forward(self, batch: tuple[Tensor, Any]) -> Tensor:
        inputs, _ = batch
        outputs = self.module(inputs)
        return outputs
