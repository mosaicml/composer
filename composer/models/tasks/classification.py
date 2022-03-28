"""A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.

:class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import Accuracy

from composer.core.types import BatchPair
from composer.loss import soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models import ComposerModel

__all__ = ["ComposerClassifier"]


class ComposerClassifier(ComposerModel):
    """A convenience class that creates a :class:`.ComposerModel` for classification tasks from a vanilla PyTorch model.
    :class:`.ComposerClassifier` requires batches in the form: (``input``, ``target``) and includes a basic
    classification training loop with :func:`.soft_cross_entropy` loss and accuracy logging.

    Args:
        module (torch.nn.Module): A PyTorch neural network module.

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

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_loss = CrossEntropy()
        self.module = module

        if hasattr(self.module, "num_classes"):
            self.num_classes = getattr(self.module, "num_classes")

    def loss(self, outputs: Any, batch: BatchPair, *args, **kwargs) -> Tensor:
        _, targets = batch
        if not isinstance(outputs, Tensor):  # to pass typechecking
            raise ValueError("Loss expects input as Tensor")
        if not isinstance(targets, Tensor):
            raise ValueError("Loss does not support multiple target Tensors")
        return soft_cross_entropy(outputs, targets, *args, **kwargs)

    def metrics(self, train: bool = False) -> Union[Metric, MetricCollection]:
        return self.train_acc if train else MetricCollection([self.val_acc, self.val_loss])

    def forward(self, batch: BatchPair) -> Tensor:
        inputs, _ = batch
        outputs = self.module(inputs)
        return outputs

    def validate(self, batch: BatchPair) -> Tuple[Any, Any]:
        _, targets = batch
        outputs = self.forward(batch)
        return outputs, targets
