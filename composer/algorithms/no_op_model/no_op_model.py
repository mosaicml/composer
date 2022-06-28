# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""NoOpModel algorithm and class."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification.accuracy import Accuracy

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.models.base import ComposerModel
from composer.utils import module_surgery

if TYPE_CHECKING:
    from composer.core.types import Batch

log = logging.getLogger(__name__)

__all__ = ['NoOpModelClass', 'NoOpModel']


class NoOpModelClass(ComposerModel):
    """Dummy model used for performance measurements.

    The :class:`.NoOpModel` algorithm uses this class to replace a :class:`torch.nn.Module`.

    Args:
        original_model (torch.nn.Module): Model to replace.
    """

    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        original_device = next(original_model.parameters()).device
        self.weights = torch.nn.Parameter(torch.Tensor([1.5]).to(original_device))
        try:
            # For classification
            self.num_classes = original_model.num_classes
        except AttributeError:
            pass

    def loss(self, outputs: torch.Tensor, batch: Batch):
        x, y = batch
        assert isinstance(y, torch.Tensor)
        del x  # unused
        return F.mse_loss(outputs, y.to(torch.float32))

    def forward(self, batch: Batch):
        x, y = batch
        del x  # unused
        assert isinstance(y, torch.Tensor)
        return y * self.weights

    def metrics(self, train: bool) -> Union[Metric, MetricCollection]:
        return Accuracy()

    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        x, y = batch
        del x  # unused
        return y, y


class NoOpModel(Algorithm):
    """Runs on :attr:`Event.INIT` and replaces the model with a dummy :class:`.NoOpModelClass` instance."""

    def __init__(self):
        # No arguments
        pass

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        new_model = NoOpModelClass(state.model)
        module_surgery.update_params_in_optimizer(old_params=state.model.parameters(),
                                                  new_params=new_model.parameters(),
                                                  optimizers=state.optimizers)
        state.model = new_model

        log.info('Replaced model with a NoOpModel')
