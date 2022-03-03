# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Metrics, Tensor, as_batch_pair
from composer.models.base import ComposerModel
from composer.utils import module_surgery

if TYPE_CHECKING:
    from composer.core.types import Batch

log = logging.getLogger(__name__)

__all__ = ["NoOpModelClass", "NoOpModel"]


class NoOpModelClass(ComposerModel):

    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor([1.5]))
        try:
            # For classification
            self.num_classes = original_model.num_classes
        except AttributeError:
            pass

    def loss(self, outputs: Tensor, batch: Batch):
        x, y = as_batch_pair(batch)
        assert isinstance(y, Tensor)
        del x  # unused
        return F.mse_loss(outputs, y.to(torch.float32))

    def forward(self, batch: Batch):
        x, y = as_batch_pair(batch)
        del x  # unused
        assert isinstance(y, Tensor)
        return y * self.weights

    def metrics(self, train: bool) -> Metrics:
        return Accuracy()

    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        x, y = as_batch_pair(batch)
        del x  # unused
        return y, y


class NoOpModel(Algorithm):

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        # replace model with dummy model
        new_model = NoOpModelClass(state.model)
        module_surgery.update_params_in_optimizer(old_params=state.model.parameters(),
                                                  new_params=new_model.parameters(),
                                                  optimizers=state.optimizers)
        state.model = new_model

        log.info('Replaced model with a NoOpModel')
