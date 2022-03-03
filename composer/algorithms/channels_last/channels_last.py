# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from typing import Optional

import torch

from composer.core.types import Algorithm, Event, Logger, State

log = logging.getLogger(__name__)


def apply_channels_last(model: torch.nn.Module) -> None:
    """Changes the memory format of the model to torch.channels_last.

    This usually yields improved GPU utilization.

    Args:
        model: model or module to modify
    """
    model.to(memory_format=torch.channels_last)  # type: ignore


class ChannelsLast(Algorithm):
    """Changes the memory format of the model to ``torch.channels_last``. This usually yields improved GPU utilization.

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
        apply_channels_last(state.model)

        log.info(f'Model {state.model.__class__.__name__} changed to channels_last format.')
