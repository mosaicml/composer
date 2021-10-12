# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
from torchmetrics.classification.accuracy import Accuracy

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Metrics, Precision
from composer.models.base import BaseMosaicModel

if TYPE_CHECKING:
    from composer.core.types import Batch, Tensors

log = logging.getLogger(__name__)


# TODO: enable for mixed precision
# TODO: enable for DDP
# TODO: enable for eval
class NoOpModelClass(BaseMosaicModel):

    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        self.parameter = torch.nn.parameter.Parameter(data=torch.tensor(0.0))
        self.zero_tensor = torch.tensor(0.0)
        try:
            # For classification
            self.num_classes = original_model.num_classes
        except AttributeError:
            pass

    def loss(self, outputs: Any, batch: Batch, *args, **kwargs) -> Tensors:
        mock_loss = self.zero_tensor + self.parameter
        return mock_loss

    def forward(self, batch: Batch) -> Tensors:
        return self.zero_tensor

    def metrics(self, train: bool) -> Metrics:
        return Accuracy()

    def validate(self, batch: Batch) -> Tuple[Any, Any]:
        raise NotImplementedError("NoOpModel not supported for eval yet.")


@dataclass
class NoOpModelHparams(AlgorithmHparams):

    def initialize_object(self) -> NoOpModel:
        return NoOpModel(**asdict(self))


class NoOpModel(Algorithm):

    def __init__(self):
        pass

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        # replace model with dummy model
        if state.precision == Precision.AMP:
            raise NotImplementedError('NoOpModel not supported for AMP Precision yet.')

        state.model = NoOpModelClass(state.model)
        log.info('Replaced model with a NoOpModel')
