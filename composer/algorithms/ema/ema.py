# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Exponential Moving Average (EMA) classes and functions."""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Dict, List, Optional, Union

import torch

from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['EMA', 'compute_ema']


def compute_ema(model: T_Model, ema_model: T_Model, smoothing: float = 0.99):
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
        half_life (str): The time string specifying the half life for terms in the average. A longer half life means
            old information is remembered longer, a shorter half life means old information is discared sooner.
            A half life of ``0`` means no averaging is done, an infinite half life means no update is done. Currently
            only units of epoch ('ep') and batch ('ba'). Value must be an integer.
        update_interval (str, optional): The time string specifying the period at which updates are done. For example,
            an ``update_interval='1ep'`` means updates are done every epoch, while ``update_interval='10ba'`` means
            updates are done once every ten batches. Units must match the units used to specify ``half_life``. If not
            specified, ``update_interval`` will default to ``1`` in the units of ``half_life``. Value must be an
            integer. Default: ``None``.
        train_with_ema_weights (bool, optional): An experimental feature that uses the ema weights as the training
            weights. In most cases should be left as ``False``. Default ``False``.

    Example:
        .. testcode::

            from composer.algorithms import EMA
            algorithm = EMA(half_life='50ba', update_interval='1ba')
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )
    """

    def __init__(self, half_life: str, update_interval: Optional[str] = None, train_with_ema_weights: bool = False):
        self.half_life = half_life
        self.update_interval = update_interval
        self.train_with_ema_weights = train_with_ema_weights

        self.ema_model = None
        self.training_model = None

        self.serialized_attributes = [
            'ema_model',
            'training_model',
        ]

        # Check timestrings are parsable and convert into time object
        try:
            self.half_life = Time.from_timestring(half_life)
        except ValueError as error:
            raise ValueError(f'Invalid time string for parameter half_life') from error

        # Create the update interval if none is specified
        if self.update_interval is None:
            self.update_interval = Time(1, self.half_life.unit)
        elif type(update_interval) is str:
            try:
                self.update_interval = Time.from_timestring(update_interval)
            except ValueError as error:
                raise ValueError(f'Invalid time string for parameter update_interval') from error
        else:
            raise ValueError(f'update_interval must be None or a time string.')

        # Verify that the units of half_life and update_interval are compatible
        if self.half_life.unit != self.update_interval.unit:
            raise ValueError(f'Units of half_life and update_interval must match.')

        # Verify that the time strings have supported units.
        if self.half_life.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter half_life: '
                             f'{self.update_interval.unit}')

        # Calculate the appropriate weighting for the moving average
        self.smoothing = 2**(-(self.update_interval.value / self.half_life.value))

        # Construct the appropriate matching events
        self.match_events = [Event.FIT_START, Event.EVAL_START, Event.EVAL_END]
        if self.half_life.unit == TimeUnit.EPOCH:
            self.match_events.append(Event.EPOCH_END)
        if self.half_life.unit == TimeUnit.BATCH:
            self.match_events.append(Event.BATCH_END)

    def match(self, event: Event, state: State) -> bool:
        return event in self.match_events

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        assert isinstance(self.update_interval, Time)

        if event == Event.FIT_START:
            if self.ema_model is not None:
                _move_shadow_model_to_device(self.ema_model, state.model)
            if self.training_model is not None:
                _move_shadow_model_to_device(self.training_model, state.model)

        if event in [Event.BATCH_END, Event.EPOCH_END]:
            # Check if an update should happen
            if state.timestamp.get(self.update_interval.unit).value % self.update_interval.value == 0:
                # Initialize the shadow models if they don't exist yet
                if self.ema_model is None:
                    self.ema_model = ShadowModel(state.model)
                if self.training_model is None and self.train_with_ema_weights is False:
                    self.training_model = ShadowModel(state.model)

                # Update the ema model
                compute_ema(state.model, self.ema_model, smoothing=self.smoothing)
                if self.train_with_ema_weights:
                    # Use the ema weights for further training
                    _copy_model(self.ema_model, state.model)

        if event == Event.EVAL_START and self.ema_model is not None and self.training_model is not None:
            # Swap out the training model for the ema model in state
            _copy_model(state.model, self.training_model)
            _copy_model(self.ema_model, state.model)

        if event == Event.EVAL_END and self.training_model is not None:
            # Swap out the ema model for the training model in state
            _copy_model(self.training_model, state.model)

    def get_ema_model(self, model: torch.nn.Module):
        """Copies ema model parameters and buffers to the input model and returns it.

        Args:
            model (torch.nn.Module): the model to convert into the ema model.

        Returns:
            torch.nn.Module: The input model with parameters and buffers replaced
                with the averaged parameters and buffers.
        """
        if self.ema_model is None:
            raise AttributeError('ema model has not been initialized yet')

        _copy_model(self.ema_model, model)
        return model

    def state_dict(self) -> Dict[str, ShadowModel]:
        state_dict = {}
        for attribute_name in self.serialized_attributes:
            shadow_model = getattr(self, attribute_name)
            state_dict[attribute_name] = {}
            state_dict[attribute_name]['parameters'] = shadow_model.parameters()
            state_dict[attribute_name]['buffers'] = shadow_model.buffers()
        return state_dict

    def load_shadow_model(self, name, parameters: List, buffers: List):
        shadow_model = ShadowModel(None)
        shadow_model.param_list = parameters
        shadow_model.buffer_list = buffers
        setattr(self, name, shadow_model)

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False):
        for attribute_name, serialized_value in state.items():
            self.load_shadow_model(attribute_name, serialized_value['parameters'], serialized_value['buffers'])


class ShadowModel:
    """A shadow model that tracks parameters and buffers from an original source model.

    Args:
        model (torch.nn.Module): the source model containing the parameters and buffers to shadow.
    """

    def __init__(self, model: Union[None, torch.nn.Module]):
        if model is not None:
            self.param_list = [copy.deepcopy(p.data) for p in model.parameters()]
            self.buffer_list = [copy.deepcopy(b.data) for b in model.buffers()]
        else:
            self.param_list = []
            self.buffer_list = []

    def parameters(self):
        return self.param_list

    def buffers(self):
        return self.buffer_list


T_Model = Union[torch.nn.Module, ShadowModel]


def _copy_model(source_model: T_Model, destination_model: T_Model):
    """Copies parameters and buffers from ``source_model`` to ``destination_model``."""
    with torch.no_grad():
        source_params = itertools.chain(source_model.parameters(), source_model.buffers())
        destination_params = itertools.chain(destination_model.parameters(), destination_model.buffers())

        for source_param, destination_param in zip(source_params, destination_params):
            destination_param.data = source_param.data


def _move_shadow_model_to_device(shadow_model: ShadowModel, destination_model: torch.nn.Module):
    """Ensures the tensors of a shadow model are on the same device as a destination model."""
    with torch.no_grad():
        destination_params = destination_model.parameters()
        shadow_params = shadow_model.parameters()
        shadow_model.param_list = [s.to(d.device) for s, d in zip(shadow_params, destination_params)]

        destination_buffers = destination_model.buffers()
        shadow_buffers = shadow_model.buffers()
        shadow_model.buffer_list = [s.to(d.device) for s, d in zip(shadow_buffers, destination_buffers)]
