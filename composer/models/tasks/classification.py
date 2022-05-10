# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.

:class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy

from composer.loss import loss_registry
from composer.metrics import CrossEntropy
from composer.models import ComposerModel

__all__ = ["ComposerClassifier"]

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

    def __init__(self, module: torch.nn.Module, loss_name: str = "soft_cross_entropy") -> None:
        super().__init__()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_loss = CrossEntropy()
        self.module = module
        if loss_name not in loss_registry.keys():
            raise ValueError(f"Unrecognized loss function: {loss_name}. Please ensure the "
                             "specified loss function is present in composer.loss.loss.py")
        self._loss_fxn = loss_registry[loss_name]

        if hasattr(self.module, "num_classes"):
            self.num_classes = getattr(self.module, "num_classes")

        if loss_name == 'binary_cross_entropy_with_logits':
            log.warning("UserWarning: Using `binary_cross_entropy_loss_with_logits` "
                        "without using `initializers.linear_log_constant_bias` can degrade "
                        "performance. "
                        "Please ensure you are using `initializers. "
                        "linear_log_constant_bias`.")

    def loss(self, outputs: Any, batch: Any, *args, **kwargs) -> Tensor:
        _, targets = batch
        if not isinstance(outputs, Tensor):  # to pass typechecking
            raise ValueError("Loss expects input as Tensor")
        if not isinstance(targets, Tensor):
            raise ValueError("Loss does not support multiple target Tensors")
        return self._loss_fxn(outputs, targets, *args, **kwargs)

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        return self.train_acc if train else MetricCollection([self.val_acc, self.val_loss])

    def forward(self, batch: Any) -> Tensor:
        inputs, _ = batch
        outputs = self.module(inputs)
        return outputs

    def validate(self, batch: Any) -> Tuple[Any, Any]:
        _, targets = batch
        outputs = self.forward(batch)
        return outputs, targets
