# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import textwrap
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery
from composer.core.types import Metrics, Precision, Tensor, as_batch_pair
from composer.models.base import ComposerModel

if TYPE_CHECKING:
    from composer.core.types import Batch

log = logging.getLogger(__name__)


class NoOpModelClass(ComposerModel):

    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        torch.autograd.set_detect_anomaly(True)
        self.weights = torch.tensor([1.5], requires_grad=True, dtype=torch.float)
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

        if not isinstance(state.model, ComposerModel):
            # We do NOT want to apply this algorithm after deepspeed or DDP wrapping
            # the module.
            # Hence, we raise an error if the model is already wrapped (i.e. it is no longer a ComposerModel)
            # when the algorithm is not yet applied
            raise RuntimeError(
                textwrap.dedent(f"""\
                Unable to apply {type(self).__qualname__} on model of type {type(state.model).__qualname__};
                expected state.model to be {ComposerModel.__qualname__}"""))
        self._applied = True

        if state.precision == Precision.AMP:
            raise NotImplementedError('NoOpModel not supported for AMP Precision yet.')

        new_model = NoOpModelClass(state.model)
        surgery.update_params_in_optimizer(old_params=state.model.parameters(),
                                           new_params=new_model.parameters(),
                                           optimizers=state.optimizers)
        state.model = new_model

        log.info('Replaced model with a NoOpModel')
