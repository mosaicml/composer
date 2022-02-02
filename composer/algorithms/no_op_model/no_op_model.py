# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import textwrap
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
from torchmetrics.classification.accuracy import Accuracy

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Metrics, Precision
from composer.models.base import ComposerModel

if TYPE_CHECKING:
    from composer.core.types import Batch, Tensors

log = logging.getLogger(__name__)


# TODO: enable for mixed precision
# TODO: enable for DDP
# TODO: enable for eval
class NoOpModelClass(ComposerModel):

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
        self._applied = False

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT and not self._applied

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        # replace model with dummy model

        if not isinstance(state.model, BaseMosaicModel):
            # We do NOT want to apply this algorithm after deepspeed or DDP wrapping
            # the module.
            # Hence, we raise an error if the model is already wrapped (i.e. it is no longer a BaseMosaicModel)
            # when the algorithm is not yet applied
            raise RuntimeError(
                textwrap.dedent(f"""\
                Unable to apply {type(self).__name__} on model of type {type(state.model)};
                expected state.model to be {BaseMosaicModel.__name__}"""))
        self._applied = True

        if state.precision == Precision.AMP:
            raise NotImplementedError('NoOpModel not supported for AMP Precision yet.')

        state.model = NoOpModelClass(state.model)
        log.info('Replaced model with a NoOpModel')
