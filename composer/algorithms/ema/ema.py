# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Exponential Moving Average (EMA) classes and functions."""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, Optional

import torch

from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['EMA', 'compute_ema']


def compute_ema(model: torch.nn.Module, ema_model: torch.nn.Module, smoothing: float = 0.99):
    r"""Updates the weights of ``ema_model`` to be closer to the weights of ``model``
    according to an exponential weighted average. Weights are updated according to

    .. math::
        W_{ema_model}^{(t+1)} = smoothing\times W_{ema_model}^{(t)}+(1-smoothing)\times W_{model}^{(t)}

    The update to ``ema_model`` happens in place.

    The half life of the weights for terms in the average is given by

    .. math::
        t_{1/2} = -\frac{\log(2)}{\log(smoothing)}

    Therefore, to set smoothing to obtain a target half life, set smoothing according to

    .. math::
        smoothing = \exp\left[-\frac{\log(2)}{t_{1/2}}\right]

    Args:
        model (torch.nn.Module): the model containing the latest weights to use to update the moving average weights.
        ema_model (torch.nn.Module): the model containing the moving average weights to be updated.
        smoothing (float, optional): the coefficient representing the degree to which older observations are kept.
            Must be in the interval :math:`(0, 1)`. Default: ``0.99``.

    Example:
        .. testcode::

                import composer.functional as cf
                from torchvision import models
                model = models.resnet50()
                ema_model = models.resnet50()
                cf.compute_ema(model, ema_model, smoothing=0.9)
    """
    with torch.no_grad():
        model_params = itertools.chain(model.parameters(), model.buffers())
        ema_model_params = itertools.chain(ema_model.parameters(), ema_model.buffers())

        for ema_param, model_param in zip(ema_model_params, model_params):
            model_param = model_param.detach()
            ema_param.copy_(ema_param * smoothing + (1. - smoothing) * model_param)


class EMA(Algorithm):
    r"""Maintains a shadow model with weights that follow the exponential moving average of the trained model weights.

    Weights are updated according to

    .. math::
        W_{ema_model}^{(t+1)} = smoothing\times W_{ema_model}^{(t)}+(1-smoothing)\times W_{model}^{(t)}

    Where the smoothing is determined from ``half_life`` according to

    .. math::
        smoothing = \exp\left[-\frac{\log(2)}{t_{1/2}}\right]

    Model evaluation is done with the moving average weights, which can result in better generalization. Because of the
    shadow models, EMA triples the model's memory consumption. Note that this does not mean that the total memory
    required doubles, since stored activations and the optimizer state are not duplicated. EMA also uses a small
    amount of extra compute to update the moving average weights.

    See the :doc:`Method Card </method_cards/ema>` for more details.

    Args:
        half_life (str, optional): The time string specifying the half life for terms in the average. A longer half
            life means old information is remembered longer, a shorter half life means old information is discared
            sooner. A half life of ``0`` means no averaging is done, an infinite half life means no update is done.
            Currently only units of epoch ('ep') and batch ('ba'). Time must be an integer value in the units
            specified. Cannot be used if ``smoothing`` is also specified. Default: ``"1000ba"``.
        smoothing (float, optional): The coefficient representing the degree to which older observations are kept.
            Must be in the interval :math:`(0, 1)`. Cannot be used if ``half_life`` also specified. This value will
            not be adjusted if ``update_interval`` is changed. Default: ``None``.
        ema_start (str, optional): The time string denoting the amount of training completed before EMA begins.
            Currently only units of duration ('dur'), batch ('ba') and epoch ('ep') are supported.
            Default: ``'0.0dur'``.
        update_interval (str, optional): The time string specifying the period at which updates are done. For example,
            an ``update_interval='1ep'`` means updates are done every epoch, while ``update_interval='10ba'`` means
            updates are done once every ten batches. Units must match the units used to specify ``half_life`` if not
            using ``smoothing``. If not specified, ``update_interval`` will default to ``1`` in the units of
            ``half_life``, or ``"1ba"`` if ``smoothing`` is specified. Time must be an integer value in the units
            specified. Default: ``None``.

    Example:
        .. testcode::

            from composer.algorithms import EMA
            algorithm = EMA(half_life='1000ba', update_interval='1ba')
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self,
                 half_life: Optional[str] = '1000ba',
                 smoothing: Optional[float] = None,
                 ema_start: str = '0.0dur',
                 update_interval: Optional[str] = None):
        self.ema_model = None
        self.training_model = None
        self.ema_weights_active = False
        self.ema_started = False
        self.serialized_attributes = ['ema_model', 'training_model', 'ema_weights_active', 'ema_started']

        # Verify that either half_life or smoothing has been specified
        if half_life is None and smoothing is None:
            raise ValueError(f'Either half_life or smoothing must be specified')

        # Verify that only one of half_life or smoothing has been specified
        if half_life is not None and smoothing is not None:
            raise ValueError(f'Only one of  half_life or smoothing can be specified')

        # Check timestrings are parsable and convert into time object
        if half_life is not None:
            self.half_life = Time.from_timestring(half_life)

        # Convert start time to a time object
        self.ema_start = Time.from_timestring(ema_start)

        # Create the update interval if none is specified
        if update_interval is None:
            if self.half_life:
                self.update_interval = Time(1, self.half_life.unit)
            else:
                self.update_interval = Time(1, TimeUnit.BATCH)
        elif type(update_interval) is str:
            self.update_interval = Time.from_timestring(update_interval)
        else:
            raise ValueError(f'update_interval must be None or a time string.')

        # Verify that the units of half_life and update_interval are compatible if necessary
        if half_life is not None and self.half_life.unit != self.update_interval.unit:
            raise ValueError(f'Units of half_life and update_interval must match.')

        # Verify that the time strings have supported units.
        if self.update_interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter update_interval: '
                             f'{self.update_interval.unit}')

        # Calculate the appropriate weighting for the moving average
        if smoothing is None and self.half_life:
            self.smoothing = 2**(-(self.update_interval.value / self.half_life.value))
        else:
            self.smoothing = smoothing

        # Construct the appropriate matching events
        self.match_events = [Event.FIT_START, Event.BATCH_START, Event.EVAL_START, Event.EVAL_END]
        self.checkpoint_events = [Event.BATCH_CHECKPOINT, Event.EPOCH_CHECKPOINT]
        if self.update_interval.unit == TimeUnit.BATCH:
            self.update_event = Event.BATCH_END
        elif self.update_interval.unit == TimeUnit.EPOCH:
            self.update_event = Event.EPOCH_END

    def _should_start(self, state: State) -> bool:
        if self.ema_start.unit == TimeUnit.DURATION:
            current_time = state.get_elapsed_duration()
            if current_time is not None:
                should_start = (self.ema_start <= current_time)
            else:
                should_start = False
        else:
            current_time = state.timestamp.get(self.ema_start.unit).value
            should_start = (self.ema_start.value <= current_time)

        return should_start

    def match(self, event: Event, state: State) -> bool:
        # Always run on init
        if event == Event.INIT:
            return True

        # Check if ema should start running, and if so reinitialize models
        if event == self.update_event and self.ema_started is False and self._should_start(state):
            self.ema_model = copy.deepcopy(state.model)
            self.training_model = copy.deepcopy(state.model)
            self.ema_started = True

        # Match on checkpointing events if a checkpoint is to be saved
        if event in [Event.BATCH_CHECKPOINT, Event.EPOCH_CHECKPOINT] and self.ema_started:
            checkpoint_savers = [cb for cb in state.callbacks if isinstance(cb, CheckpointSaver)]
            for checkpoint_saver in checkpoint_savers:
                if checkpoint_saver.save_interval(state, event) is True:
                    return True

        # Otherwise, always run on some events after ema has started
        if event in self.match_events and self.ema_started:
            return True

        # Conditionally run on the update event if ema has started
        if event == self.update_event and self.ema_started:
            return (state.timestamp.get(self.update_interval.unit).value % self.update_interval.value == 0)

        return False

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        assert isinstance(self.update_interval, Time)
        assert isinstance(self.smoothing, float)

        if event == Event.INIT:
            # Create the models so that the checkpoints can be loaded
            self.ema_model = copy.deepcopy(state.model)
            self.training_model = copy.deepcopy(state.model)

        assert self.ema_model is not None
        assert self.training_model is not None

        if event == Event.FIT_START:
            # Ensure that params are on the right device if a checkpoint has been loaded
            _move_params_to_device(model=self.ema_model, destination_model=state.model)
            _move_params_to_device(model=self.training_model, destination_model=state.model)

        if event == Event.BATCH_START and self.ema_weights_active:
            # Ensure the model being trained has the correct weights
            _copy_params(source_model=self.training_model, destination_model=state.model)
            self.ema_weights_active = False

        if event in [Event.BATCH_END, Event.EPOCH_END]:
            # Update the ema model
            compute_ema(state.model, self.ema_model, smoothing=self.smoothing)

        if event == Event.EVAL_START and self.ema_weights_active is False:
            # Swap out the training model for the ema model in state
            _copy_params(source_model=state.model, destination_model=self.training_model)
            _copy_params(source_model=self.ema_model, destination_model=state.model)
            self.ema_weights_active = True

        if event == Event.EVAL_END:
            # Swap out the ema model for the training model in state
            _copy_params(source_model=self.training_model, destination_model=state.model)
            self.ema_weights_active = False

        if event in self.checkpoint_events and self.ema_weights_active is False:
            # Swap the training model out for the ema model for checkpointing
            _copy_params(source_model=state.model, destination_model=self.training_model)
            _copy_params(source_model=self.ema_model, destination_model=state.model)
            self.ema_weights_active = True

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        for attribute_name in self.serialized_attributes:
            if attribute_name in ['ema_model', 'training_model']:
                model = getattr(self, attribute_name)
                state_dict[attribute_name] = model.state_dict()
            else:
                state_dict[attribute_name] = getattr(self, attribute_name)
        return state_dict

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False):
        for attribute_name, serialized_value in state.items():
            if attribute_name != 'repr':  # skip attribute added by parent class
                if attribute_name == 'ema_model' and self.ema_model is not None:
                    self.ema_model.load_state_dict(serialized_value)
                elif attribute_name == 'training_model' and self.training_model is not None:
                    self.training_model.load_state_dict(serialized_value)
                else:
                    setattr(self, attribute_name, serialized_value)


def _copy_params(source_model: torch.nn.Module, destination_model: torch.nn.Module):
    """Copies parameters and buffers from ``source_model`` to ``destination_model``."""
    with torch.no_grad():
        source_params = itertools.chain(source_model.parameters(), source_model.buffers())
        destination_params = itertools.chain(destination_model.parameters(), destination_model.buffers())

        for source_param, destination_param in zip(source_params, destination_params):
            destination_param.data = source_param.data


def _move_params_to_device(model: torch.nn.Module, destination_model: torch.nn.Module):
    """Ensures the parameters of a model are on the same device as a destination model."""
    with torch.no_grad():
        destination_params = destination_model.parameters()
        params = model.parameters()
        for s, d in zip(params, destination_params):
            s.to(d.device)

        destination_buffers = destination_model.buffers()
        buffers = model.buffers()
        for s, d in zip(buffers, destination_buffers):
            s.to(d.device)
