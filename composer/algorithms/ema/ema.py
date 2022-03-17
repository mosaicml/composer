# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Exponential Moving Average (EMA) classes and functions."""

import copy
import logging
from typing import List, Optional, Tuple

import torch

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Model, Optimizers

log = logging.getLogger(__name__)

__all__ = ["EMA", "ema"]


def ema(model: Model, ema_model: Model, alpha: float = 0.9):
    """
    The half life of the weights for terms in the average is given by

    .. math::
        t_{1/2} = -\frac{\log(2)}{\log(\alpha)}

    Therefore to set alpha to obtain a target half life, set alpha according to
    .. math::
        \alpha = \exp\left[- \frac{\log(2)}{t_{1/2}}\right]

    Args:
        model (Model): _description_
        ema_model (Model): _description_
        alpha (float, optional): _description_. Defaults to 0.9.
    """
    model_dict = model.state_dict()
    for key, ema_param in ema_model.state_dict().items():
        model_param = model_dict[key].detach()
        ema_param.copy_(ema_param * alpha + (1. - alpha) * model_param)


class EMA(Algorithm):
    """
    """

    def __init__(self, alpha: float, update_interval: str, train_with_ema_weights: bool):
        self.alpha = alpha
        self.update_interval = update_interval
        self.train_with_ema_weights = train_with_ema_weights

        self.ema_model = None
        self.training_model = None

        # Check update_interval timestring is parsable and convert into time object
        try:
            self.update_interval = Time.from_timestring(update_interval)
        except ValueError as error:
            raise ValueError(f"Invalid time string for parameter update_interval") from error

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.FIT_START, Event.BATCH_END, Event.EVAL_START, Event.EVAL_END]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.FIT_START:
            # Initialize the ema model
            self.ema_model = copy.deepcopy(state.model)
            self.training_model = copy.deepcopy(state.model)

        if self.train_with_ema_weights:
            if event == Event.BATCH_END:
                ema(state.model, self.ema_model, alpha=self.alpha)
                state.model.load_state_dict(self.ema_model.state_dict())
        else:
            if event == Event.BATCH_END:
                # Update the ema model
                ema(state.model, self.ema_model, alpha=self.alpha)
            if event == Event.EVAL_START:
                # Swap out the training model for the ema model in state
                self.training_model.load_state_dict(state.model.state_dict())
                state.model.load_state_dict(self.ema_model.state_dict())
            if event == Event.EVAL_END:
                # Swap out the ema model for the training model in state
                state.model.load_state_dict(self.training_model.state_dict())
