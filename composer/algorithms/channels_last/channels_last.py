# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)


class ChannelsLast(Algorithm):
    """Changes the memory format of the model to ``torch.channels_last``.
    This usually yields improved GPU utilization.

    Runs on ``Event.INIT``, so it can set the memory format before the model is DDP wrapped. Has no hyperparameters.
    """

    def match(self, event: Event, state: State) -> bool:
        """Runs on ``Event.INIT``"""
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Changes the memory format of the model to ``torch.channels_last``"""
        del event, logger  # unused
        # TODO: Double check model is moved to cuda with device type
        state.model.to(memory_format=torch.channels_last)  # type: ignore

        log.info(f'Model {state.model.__class__.__name__} changed to channels_last format.')


@dataclass
class ChannelsLastHparams(AlgorithmHparams):
    """ChannelsLast has no hyperparameters, so this class has no member variables"""
    pass

    def initialize_object(self) -> ChannelsLast:
        return ChannelsLast()
