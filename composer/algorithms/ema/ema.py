# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Exponential Moving Average (EMA) classes and functions."""

import copy
import logging
from typing import List, Optional, Tuple

import torch

from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger

import pdb
log = logging.getLogger(__name__)

__all__ = ["EMA", "ema"]


def ema(model: torch.nn.Module, ema_model: torch.nn.Module, alpha: float = 0.9):
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

    def __init__(self, half_life: str, update_interval: str, train_with_ema_weights: bool):
        self.half_life = half_life
        self.update_interval = update_interval
        self.train_with_ema_weights = train_with_ema_weights

        self.ema_model = None
        self.training_model = None

        # Check timestrings are parsable and convert into time object
        try:
            self.half_life = Time.from_timestring(half_life)
        except ValueError as error:
            raise ValueError(f"Invalid time string for parameter half_life") from error

        try:
            self.update_interval = Time.from_timestring(update_interval)
        except ValueError as error:
            raise ValueError(f"Invalid time string for parameter update_interval") from error

        # Verify that the units of half_life and update_interval are compatible
        if self.half_life.unit != self.update_interval.unit:
            raise ValueError(f"Units of half_life and update_interval must match.")

        # Verify that the time strings have supported units.
        if self.half_life.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f"Invalid unit string for parameter half_life: "
                             f"{self.update_interval.unit}")

        # Calculate the appropriate weighting for the moving average
        self.alpha = 2 ** (-(self.update_interval.value/self.half_life.value))

        # Construct the appropriate matching events
        self.match_events = [Event.FIT_START, Event.EVAL_START, Event.EVAL_END]
        if self.half_life.unit == TimeUnit.EPOCH:
            self.match_events.append(Event.EPOCH_END)
        if self.half_life.unit == TimeUnit.BATCH:
            self.match_events.append(Event.BATCH_END)

    def match(self, event: Event, state: State) -> bool:
        return event in self.match_events

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.FIT_START:
            # Initialize the ema model
            self.ema_model = copy.deepcopy(state.model)
            self.training_model = copy.deepcopy(state.model)

        if self.train_with_ema_weights:
            if event in [Event.BATCH_END, Event.EPOCH_END]:
                # Check if an update should happen
                if state.timer.get(self.update_interval.unit).value % self.update_interval.value == 0:
                    # Update the ema model
                    ema(state.model, self.ema_model, alpha=self.alpha)
                    # Use the ema weights for further training
                    state.model.load_state_dict(self.ema_model.state_dict())
        else:
            if event in [Event.BATCH_END, Event.EPOCH_END]:
                # Check if an update should happen
                if state.timer.get(self.update_interval.unit).value % self.update_interval.value == 0:
                    print(state.timer.get(self.update_interval.unit), "Updating", self.alpha)
                    # Update the ema model
                    ema(state.model, self.ema_model, alpha=self.alpha)
            if event == Event.EVAL_START:
                # Swap out the training model for the ema model in state
                self.training_model.load_state_dict(state.model.state_dict())
                state.model.load_state_dict(self.ema_model.state_dict())
            if event == Event.EVAL_END:
                # Swap out the ema model for the training model in state
                state.model.load_state_dict(self.training_model.state_dict())
