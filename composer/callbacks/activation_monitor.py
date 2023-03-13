# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor activation values during training."""

from functools import partial
from typing import List, Optional, Sequence, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger

__all__ = ['ActivationMonitor']


class ActivationMonitor(Callback):

    def __init__(self,
                 recompute_attention_softmax: bool = True,
                 interval: Union[int, str, Time] = '100ba',
                 ignore_module_types: Optional[List[str]] = []):
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

            self.attach_forward_hook(state, logger)

    def after_forward(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.interval.unit).value

        if current_time_value % self.interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self.remove_forward_hooks()

    def attach_forward_hook(self, state: State, logger: Logger):
        self.register_forward_hook(state.model, logger)

    def remove_forward_hooks(self):
        for handle in self.handles:
            handle.remove()
        # Resetting handles we track
        self.handles = []

    def register_forward_hook(self, model: torch.nn.Module, logger: Logger):
        model.apply(partial(self._register_forward_hook, logger))

    def _register_forward_hook(self, logger: Logger, module: torch.nn.Module):
        self.handles.append(module.register_forward_hook(partial(self.forward_hook, logger)))

    def forward_hook(self, logger: Logger, module: torch.nn.Module, input: Sequence, output: Sequence):
        module_name = self.module_names[module]

        for ignore_module_type in self.ignore_module_types:
            if ignore_module_type in module_name:
                return

        metric_name = f'activations/{module_name}'
        metrics = {}

        for i, val in enumerate(input):
            if val is None or isinstance(val, dict):
                continue
            self.add_metrics(metrics, f'{metric_name}_input_{i}', val)

        for i, val in enumerate(output):
            if val is None or isinstance(val, dict):
                continue
            self.add_metrics(metrics, f'{metric_name}_output_{i}', val)

        logger.log_metrics(metrics)

    def create_module_names(self, model: torch.nn.Module):
        self.module_names = {m: name for name, m in model.named_modules()}

    def add_metrics(self, metrics: dict, name: str, value: torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            metrics[f'{name}_l2_norm'] = torch.linalg.vector_norm(value).item()
            metrics[f'{name}_average'] = value.mean().item()
        metrics[f'{name}_max'] = value.max().item()
