# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ChannelsLast algorithm."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['ChannelsLast', 'apply_channels_last']


def apply_channels_last(model: torch.nn.Module) -> None:
    """Changes the memory format of the model to `torch.channels_last <https://\\
    pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_.

    This usually yields improved GPU utilization.

    Args:
        model (torch.nn.Module): The model or module to modify.
    """
    model.to(memory_format=torch.channels_last)  # type: ignore


class ChannelsLast(Algorithm):
    """Changes the memory format of the model to `torch.channels_last <https://\\
    pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_. This usually improves GPU utilization.

    Runs on :attr:`.Event.INIT``, so it can set the memory format before the model is DDP wrapped.
    Has no hyperparameters.

    Example:
        .. testcode::

            from composer.algorithms import ChannelsLast
            algorithm = ChannelsLast()
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self):
        # ChannelsLast takes no arguments
        pass

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        # TODO: Double check model is moved to cuda with device type
        apply_channels_last(state.model)

        log.info(f'Model {state.model.__class__.__name__} changed to channels_last format.')
