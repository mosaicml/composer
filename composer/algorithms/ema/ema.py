# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core Exponential Moving Average (EMA) classes and functions."""

from __future__ import annotations

import contextlib
import itertools
import logging
from typing import Any, Optional, Union

import torch

import composer.utils.misc as misc
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.core import Algorithm, Event, State, Time, TimeUnit
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['EMA', 'compute_ema']


def compute_ema(
    model: torch.nn.Module,
    ema_model: Union[torch.nn.Module, EMAParameters],
    smoothing: float = 0.99,
) -> None:
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
        ema_model (torch.nn.Module, EMAParameters): the model containing the moving average weights to be updated.
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
    model_context_manager = get_model_context_manager(model)

    with model_context_manager:
        with torch.no_grad():
            # If the ema model is a pytorch module, can just use the state_dict
            if isinstance(ema_model, torch.nn.Module):
                ema_params = ema_model.state_dict()
                for name, param in itertools.chain(model.named_parameters(), model.named_buffers()):
                    if name in ema_params:
                        ema_params[name].copy_(ema_params[name] * smoothing + param.data * (1. - smoothing))
            # Otherwise, the ema model needs to define the named_parameters and named_buffers dictionaries
            # These should contain the parameters and buffers to average.
            elif isinstance(ema_model, EMAParameters):
                ema_parameters = ema_model.named_parameters_dict
                ema_buffers = ema_model.named_buffers_dict
                for name, param in itertools.chain(model.named_parameters(), model.named_buffers()):
                    if name in ema_parameters:
                        ema_parameters[name].copy_(ema_parameters[name] * smoothing + param.data * (1. - smoothing))
                    if name in ema_buffers:
                        ema_buffers[name].copy_(ema_buffers[name] * smoothing + param.data * (1. - smoothing))
            else:
                raise ValueError('ema_model must be a torch.nn.Module or EMAParameters')


def get_model_context_manager(model: torch.nn.Module):
    """Summons full params for FSDP, which is required to update sharded params."""
    fsdp_enabled = misc.is_model_fsdp(model)
    model_context_manager = contextlib.nullcontext()
    if fsdp_enabled:
        model_context_manager = model.module.summon_full_params(model.module)  # type: ignore
    return model_context_manager


class EMA(Algorithm):
    r"""Maintains a set of weights that follow the exponential moving average of the training model weights.

    Weights are updated according to

    .. math::
        W_{ema_model}^{(t+1)} = smoothing\times W_{ema_model}^{(t)}+(1-smoothing)\times W_{model}^{(t)}

    Where the smoothing is determined from ``half_life`` according to

    .. math::
        smoothing = \exp\left[-\frac{\log(2)}{t_{1/2}}\right]

    Model evaluation is done with the moving average weights, which can result in better generalization. Because of the
    ema weights, EMA can double the model's memory consumption. Note that this does not mean that the total memory
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

    def __init__(
        self,
        half_life: Optional[str] = '1000ba',
        smoothing: Optional[float] = None,
        ema_start: str = '0.0dur',
        update_interval: Optional[str] = None,
    ):
        self.ema_model = None
        self.ema_weights_active = False
        self.ema_started = False
        self.serialized_attributes = ['ema_model', 'ema_weights_active', 'ema_started']

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
            raise ValueError(
                f'Invalid time unit for parameter update_interval: '
                f'{self.update_interval.unit}',
            )

        # Calculate the appropriate weighting for the moving average
        if smoothing is None and self.half_life:
            self.smoothing = 2**(-(self.update_interval.value / self.half_life.value))
        else:
            self.smoothing = smoothing

        # Construct the appropriate matching events
        self.move_device_events = [Event.EVAL_START, Event.FIT_START, Event.PREDICT_START]
        self.move_param_events = [Event.BATCH_START, Event.EVAL_START, Event.EVAL_END]
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

    def _ensure_training_weights_active(self, state: State):
        if self.ema_weights_active is True and self.ema_model is not None:
            self.ema_model.swap_params(model=state.model)
            self.ema_weights_active = False

    def _ensure_ema_weights_active(self, state: State):
        if self.ema_weights_active is False and self.ema_model is not None:
            self.ema_model.swap_params(model=state.model)
            self.ema_weights_active = True

    def match(self, event: Event, state: State) -> bool:
        # Always run on init
        if event == Event.INIT:
            return True

        # Check if ema should start running, and if so reinitialize models
        if event == self.update_event and self.ema_started is False and self._should_start(state):
            self.ema_model = EMAParameters(state.model)
            self.ema_started = True

        # Match on checkpointing events if a checkpoint is to be saved
        if event in [Event.BATCH_CHECKPOINT, Event.EPOCH_CHECKPOINT] and self.ema_started:
            checkpoint_savers = [cb for cb in state.callbacks if isinstance(cb, CheckpointSaver)]
            for checkpoint_saver in checkpoint_savers:
                assert callable(checkpoint_saver.save_interval)
                if checkpoint_saver.save_interval(state, event) is True:
                    return True

        # Otherwise, always run on events where ema params must be moved after ema has started
        if event in self.move_param_events and self.ema_started:
            return True

        # Run on events where ema params must be moved to the correct device
        if event in self.move_device_events and self.ema_started:
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
            self.ema_model = EMAParameters(state.model)

        assert self.ema_model is not None

        if event == Event.FIT_START or event == Event.PREDICT_START:
            # Ensure that params are on the right device if a checkpoint has been loaded
            self.ema_model.move_params_to_device(destination_model=state.model)

        if event == Event.BATCH_START and self.ema_weights_active:
            # Ensure the model being trained has the correct weights
            self._ensure_training_weights_active(state)

        if event in [Event.BATCH_END, Event.EPOCH_END]:
            # Update the ema model
            compute_ema(state.model, self.ema_model, smoothing=self.smoothing)

        if event == Event.EVAL_START:
            # Verify that the ema params are on the correct device.
            # Needed to ensure doing eval before training can resume correctly.
            self.ema_model.move_params_to_device(destination_model=state.model)
            # Swap out the training model for the ema model in state
            self._ensure_ema_weights_active(state)

        if event == Event.EVAL_END:
            # Swap out the ema model for the training model in state
            self._ensure_training_weights_active(state)

        if event in self.checkpoint_events:
            # Swap the training model out for the ema model for checkpointing
            self._ensure_ema_weights_active(state)

    def state_dict(self) -> dict[str, Any]:
        state_dict = super().state_dict()
        for attribute_name in self.serialized_attributes:
            if attribute_name == 'ema_model':
                ema_model = getattr(self, attribute_name)
                state_dict[attribute_name] = {}
                state_dict[attribute_name]['named_parameters_dict'] = ema_model.named_parameters_dict
                state_dict[attribute_name]['named_buffers_dict'] = ema_model.named_buffers_dict
            else:
                state_dict[attribute_name] = getattr(self, attribute_name)
        return state_dict

    def ensure_compatible_state_dict(self, state: dict[str, Any]):
        """Ensure state dicts created prior to Composer 0.13.0 are compatible with later versions."""
        # Version 0.13.0 and later state dicts will not include training_model.
        if 'training_model' not in state:
            return state

        # Prior to version 0.13.0, the state dict contained a separate training_model and ema_model.
        # Only one of these needs to be loaded as the ema_model.
        if state['ema_weights_active'] is True:
            # If EMA weights are active, load training weights into the ema_model storage
            state_dict = state['training_model']
        else:
            # If EMA weights are not active, load the ema weights into the ema_model storage
            state_dict = state['ema_model']

        named_parameters_dict = {}
        named_buffers_dict = {}
        # Rewrite the state dict in the newer format.
        if isinstance(self.ema_model, EMAParameters):
            for key in self.ema_model.named_parameters_dict.keys():
                if key in state_dict:
                    named_parameters_dict[key] = state_dict[key]
            for key in self.ema_model.named_buffers_dict.keys():
                if key in state_dict:
                    named_buffers_dict[key] = state_dict[key]
        else:
            ValueError(f'ema_model must be initialized before loading state dicts from versions earlier than 0.13.0')

        # Update the state dict with the new format
        del state['training_model']
        state['ema_model'] = {}
        state['ema_model']['named_parameters_dict'] = named_parameters_dict
        state['ema_model']['named_buffers_dict'] = named_buffers_dict
        return state

    def load_state_dict(self, state: dict[str, Any], strict: bool = False):
        state_dict = self.ensure_compatible_state_dict(state)
        for attribute_name, serialized_value in state_dict.items():
            if attribute_name != 'repr':  # skip attribute added by parent class
                if attribute_name == 'ema_model':
                    self.ema_model = EMAParameters(None)
                    self.ema_model.named_parameters_dict = serialized_value['named_parameters_dict']
                    self.ema_model.named_buffers_dict = serialized_value['named_buffers_dict']
                else:
                    setattr(self, attribute_name, serialized_value)

    def get_ema_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Replaces the parameters of the supplied model with the ema parameters if they are not already active.

        Args:
            model (torch.nn.Module): The model to replace the parameters of.

        Returns:
            torch.nn.Module: The model with the ema parameters.
        """
        assert self.ema_model is not None
        # Ensure that self.ema_model contains the ema weights. If not raise an error.
        if self.ema_weights_active == True:
            raise ValueError('The ema weight are currently contained in the composer model.')
        self.ema_model.transfer_ema_params(model=model)
        return model

    def get_training_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Replaces the parameters of the supplied model with the training parameters if they are not already active.

        Args:
            model (torch.nn.Module): The model to replace the parameters of.

        Returns:
            torch.nn.Module: The model with the training parameters.
        """
        assert self.ema_model is not None
        # Ensure that self.ema_model contains the training weights. If not raise an error.
        if self.ema_weights_active == False:
            raise ValueError('The training weights are currently contained in the composer model.')
        self.ema_model.transfer_ema_params(model=model)
        return model


class EMAParameters:
    """A class that stores the parameters and buffers of a model needed for averaging."""

    def __init__(self, model: Union[None, torch.nn.Module]):
        if model is not None:
            model_context_manager = get_model_context_manager(model)
            with model_context_manager:
                # Copy the trainable parameters and buffers.
                self.named_parameters_dict = {
                    name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad
                }
                self.named_buffers_dict = {name: buffer.data.clone() for name, buffer in model.named_buffers()}
        else:
            # Empty storage
            self.named_parameters_dict = {}
            self.named_buffers_dict = {}

    def named_parameters(self):
        return self.named_parameters_dict.items()

    def named_buffers(self):
        return self.named_buffers_dict.items()

    def swap_params(self, model: torch.nn.Module):
        """Swaps the parameters and buffers of a model with the ema parameters."""
        model_context_manager = get_model_context_manager(model)

        with torch.no_grad():
            ema_params = self.named_parameters_dict
            ema_buffers = self.named_buffers_dict

            with model_context_manager:
                for name, param in model.named_parameters():
                    if name in ema_params:
                        # Use copy instead of raw data access (eg .data) doesn't work with FSDP
                        dummy_param = param.clone()
                        param.copy_(ema_params[name])
                        ema_params[name].copy_(dummy_param)

                for name, buffer in model.named_buffers():
                    if name in ema_buffers:
                        # Use copy instead of raw data access (eg .data) doesn't work with FSDP
                        dummy_buffer = buffer.clone()
                        buffer.copy_(ema_buffers[name])
                        ema_buffers[name].copy_(dummy_buffer)

    def transfer_ema_params(self, model: torch.nn.Module):
        """Transfers the parameters and buffers from the ema model to the supplied model."""
        model_context_manager = get_model_context_manager(model)

        with model_context_manager:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.named_parameters_dict:
                        param.copy_(self.named_parameters_dict[name])

                for name, buffer in model.named_buffers():
                    if name in self.named_buffers_dict:
                        buffer.copy_(self.named_buffers_dict[name])

    def move_params_to_device(self, destination_model: torch.nn.Module):
        """Moves the ema parameters and buffers to the device of a destination model."""
        model_context_manager = get_model_context_manager(destination_model)

        with model_context_manager:
            for name, param in destination_model.named_parameters():
                if name in self.named_parameters_dict:
                    self.named_parameters_dict[name] = self.named_parameters_dict[name].to(param.device)

            for name, buffer in destination_model.named_buffers():
                if name in self.named_buffers_dict:
                    self.named_buffers_dict[name] = self.named_buffers_dict[name].to(buffer.device)
