# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Exponential Moving Average classes and functions."""

import copy
import logging
from typing import List, Optional, Tuple

import torch

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Model, Optimizers

log = logging.getLogger(__name__)

__all__ = ["ExponentialMovingAverage", "exponential_moving_average"]


def exponential_moving_average(model: Model, ema_model: Model, alpha: float) -> Model:
    """Updates an exponentially weighted moving average of the model parameters"""
    for key, param in ema_model.state_dict().items():
        ema_model.state_dict()[key].data = alpha * param.data + (1 - alpha) * model.state_dict()[key].data
    return ema_model


class ExponentialMovingAverage(Algorithm):
    """
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema_model = None
        self.training_model = None

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.FIT_START, Event.BATCH_END, Event.EVAL_START, Event.EVAL_END]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.FIT_START:
            # Initialize the ema model
            self.ema_model = copy.deepcopy(state.model)
        if event == Event.BATCH_END:
            # Update the ema model
            #self.ema_model = exponential_moving_average(state.model, self.ema_model, self.alpha)
            pass
        if event == Event.EVAL_START:
            # Swap out the training model for the ema model in state
            self.training_model = state.model
            state.model = self.ema_model
        if event == Event.EVAL_END:
            print(state.model)
            # Swap out the ema model for the training model in state
            state.model = self.training_model
