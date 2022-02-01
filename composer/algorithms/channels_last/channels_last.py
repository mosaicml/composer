# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State
from composer.models.base import BaseMosaicModel

log = logging.getLogger(__name__)


class ChannelsLast(Algorithm):
    """Changes the memory format of the model to ``torch.channels_last``.
    This usually yields improved GPU utilization.

    Runs on ``Event.INIT``, so it can set the memory format before the model is DDP wrapped. Has no hyperparameters.
    """
    def __init__(self) -> None:
        self._applied = False

    def match(self, event: Event, state: State) -> bool:
        """Runs on ``Event.INIT``"""
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Changes the memory format of the model to ``torch.channels_last``"""
        del event, logger  # unused
        if self._applied:
            return
        if not isinstance(state.model, BaseMosaicModel) and not self._applied:
            # We do NOT want to apply this algorithm after deepspeed or DDP wrapping
            # the module.
            # Hence, we raise an error if the model is already wrapped (i.e. it is no longer a BaseMosaicModel)
            # when the algorithm is not yet applied
            raise RuntimeError(f"Unable to apply {type(self).__name__} on model of type {type(state.model)}; expected state.model to be {BaseMosaicModel.__name__}")
        self._applied = True
        
        # `.to()` is missing the `memory_format` parameter in its type annotation; hence the type ignore
        # error: No overloads for "to" match the provided arguments Argument types: (memory_format)
        state.model = state.model.to(memory_format=torch.channels_last)  # type: ignore 
        log.info(f'Model {state.model.__class__.__name__} changed to channels_last format.')


@dataclass
class ChannelsLastHparams(AlgorithmHparams):
    """ChannelsLast has no hyperparameters, so this class has no member variables"""
    pass

    def initialize_object(self) -> ChannelsLast:
        return ChannelsLast()
