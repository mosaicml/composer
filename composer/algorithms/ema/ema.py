# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Exponential Moving Average (EMA) classes and functions."""

import copy
import logging
from typing import List, Optional, Tuple, Union

import torch

from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger
from composer.models import ComposerModel

log = logging.getLogger(__name__)

__all__ = ["EMA", "ema"]


def ema(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float = 0.99):
    """Updates the weights of ``ema_model`` to be closer to the weights of ``model`` according to an exponential
    weighted average. Weights are updated according to
    .. math::
        W_{ema_model}^{(t+1)} = decay\times W_{ema_model}^{(t)}+(1-decay)\times W_{model}^{(t)}
    The update to ``ema_model`` happens in place.

    The half life of the weights for terms in the average is given by
    .. math::
        t_{1/2} = -\frac{\log(2)}{\log(decay)}
    Therefore to set decay to obtain a target half life, set decay according to
    .. math::
        decay = \exp\left[- \frac{\log(2)}{t_{1/2}}\right]

    Args:
        model (torch.nn.Module): the model containing the latest weights to use to update the moving average weights.
        ema_model (torch.nn.Module): the model containing the moving average weights to be updated.
        decay (float, optional): the coefficient representing the degree to which older observations are discounted.
            Must be in the interval :math:`(0, 1)`. ``Default: ``0.99``.

    Example:
    .. testcode::
        import composer.functional as cf
        from torchvision import models
        model = models.resnet50()
        ema_model = models.resnet50()
        cf.ema(model, ema_model, decay=0.9)
    """
    model_dict = model.state_dict()
    for key, ema_param in ema_model.state_dict().items():
        model_param = model_dict[key].detach()
        ema_param.copy_(ema_param * alpha + (1. - alpha) * model_param)


class EMA(Algorithm):
    """Maintains a shadow model with weights that follow the exponential moving average of the trained model weights.

    Weights are updated according to
    .. math::
        W_{ema_model}^{(t+1)} = decay\times W_{ema_model}^{(t)}+(1-decay)\times W_{model}^{(t)}
    Where the decay is determined from ``half_life`` according to
    .. math::
        decay = \exp\left[- \frac{\log(2)}{t_{1/2}}\right]

    Model evaluation is done with the moving average weights, which can result in better generalization. Because of the
    shadow model, EMA doubles the model's memory consumption. Note that this does not mean that the total memory
    required doubles, however, since stored activations and the optimizer state are not doubled. EMA also uses a small
    amount of extra compute to update the moving average weights.

    See the :doc:`Method Card </method_cards/ema>` for more details.

    Args:
        half_life (str): The time string specifying the half life for terms in the average. A longer half life means
            old information is remembered longer, a shorter half life means old information is discared sooner.
            A half life of ``0`` means no averaging is done, an infinite half life means no update is done. Currently
            only units of epoch ('ep') and batch ('ba').
        update_interval (str): The time string specifying the period at which updates are done. For example, an
            ``update_interval='1ep'`` means updates are done every epoch, while ``update_interval='10ba'`` means
            updates are done once every ten batches. Units must match the units used to specify ``half_life``
        train_with_ema_weights (bool, optional): An experimental feature that uses the ema weights as the training
            weights. Default ``False``.
    """

    def __init__(self, half_life: str, update_interval: str, train_with_ema_weights: bool = False):
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
