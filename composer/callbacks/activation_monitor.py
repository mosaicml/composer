# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor activation values during training."""

from functools import partial
from typing import List, Optional, Sequence, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.wandb_logger import WandBLogger

__all__ = ['ActivationMonitor']


class ActivationMonitor(Callback):

    def __init__(self,
                 recompute_attention_softmax: bool = True,
                 interval: Union[int, str, Time] = '100ba',
                 ignore_module_types: Optional[List[str]] = None,
                 only_log_wandb: bool = True):
        """Logs stats of activation inputs and outputs.

        This callback triggers at a user defined interval, and logs some simple statistics of the inputs, outputs for every
        torch module. This is done by attaching a forward hook to the module. Additionally, when we are not logging
        we detach the forwards hook.

        Example:
            .. doctest::

                >>> from composer import Trainer
                >>> from composer.callbacks import ActivationMonitor
                >>> # constructing trainer object with this callback
                >>> trainer = Trainer(
                ...     model=model,
                ...     train_dataloader=train_dataloader,
                ...     eval_dataloader=eval_dataloader,
                ...     optimizers=optimizer,
                ...     max_duration="1ep",
                ...     callbacks=[ActivationMonitor()],
                ... )

        The metrics are logged by the :class:`.Logger` to the following keys described below. For convenience we have included
        example metrics logged below:

        """

        self.interval = interval
        self.recompute_attention_softmax = recompute_attention_softmax
        self.ignore_module_types = ignore_module_types
        self.only_log_wandb = only_log_wandb

        self.handles = []

        # Check that the interval timestring is parsable and convert into time object
        if isinstance(interval, int):
            self.interval = Time(interval, TimeUnit.BATCH)
        if isinstance(interval, str):
            self.interval = Time.from_timestring(interval)

        # Verify that the interval has supported units
        if self.interval.unit not in [TimeUnit.BATCH, TimeUnit.EPOCH]:
            raise ValueError(f'Invalid time unit for parameter interval: '
                             f'{self.interval.unit}')

        self.last_train_time_value_logged = -1
        self.module_names = {}

    def before_forward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            if not self.module_names:
                self.create_module_names(state.model)

            self.attach_forward_hooks(state, logger)

    def after_forward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self.remove_forward_hooks()

    def attach_forward_hooks(self, state: State, logger: Logger):
        step = state.timestamp.batch.value
        self.register_forward_hook(state.model, logger, step)

    def remove_forward_hooks(self):
        for handle in self.handles:
            handle.remove()
        # Resetting handles we track
        self.handles = []

    def register_forward_hook(self, model: torch.nn.Module, logger: Logger, step: Optional[int]):
        model.apply(partial(self._register_forward_hook, logger, step))

    def _register_forward_hook(self, logger: Logger, step: Optional[int], module: torch.nn.Module):
        self.handles.append(module.register_forward_hook(partial(self.forward_hook, logger, step)))

    def forward_hook(self, logger: Logger, step: Optional[int], module: torch.nn.Module, input: Sequence,
                     output: Sequence):
        module_name = self.module_names[module]

        if self.ignore_module_types is not None:
            for ignore_module_type in self.ignore_module_types:
                if ignore_module_type in module_name:
                    return

        metrics = {}

        for i, val in enumerate(input):
            if val is None or isinstance(val, dict):
                continue
            self.add_metrics(metrics, module_name, f'_input_{i}', val)

        for i, val in enumerate(output):
            if val is None or isinstance(val, dict):
                continue
            self.add_metrics(metrics, module_name, f'_output_{i}', val)

        if self.only_log_wandb:
            wandb_logger = [ld for ld in logger.destinations if isinstance(ld, WandBLogger)][0]
            wandb_logger.log_metrics(metrics, step)
        else:
            logger.log_metrics(metrics)

    def add_metrics(self, metrics: dict, name: str, suffix: str, value: torch.Tensor):
        # We shouldn't log booleans
        if value.dtype == torch.bool:
            return
        if value.is_floating_point() or value.is_complex():
            metrics[f'activations/l2_norm/{name}{suffix}'] = torch.linalg.vector_norm(value).item()
            metrics[f'activations/average/{name}{suffix}'] = value.mean().item()
        metrics[f'activations/max/{name}{suffix}'] = value.max().item()

    def create_module_names(self, model: torch.nn.Module):
        self.module_names = {m: name for name, m in model.named_modules()}
