# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor activation values during training."""

from functools import partial
from typing import Any, List, Optional, Sequence, Union

import torch

from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.wandb_logger import WandBLogger

__all__ = ['ActivationMonitor']


class ActivationMonitor(Callback):
    """Logs stats of activation inputs and outputs.

    This callback triggers at a user defined interval, and logs some simple statistics of the inputs, outputs for every
    torch module. This is done by attaching a forward hook to the module. Additionally, when after we finish logging
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

        +-------------------------------------------------------+-----------------------------------------------------+
        | Key                                                   | Logged data                                         |
        +=======================================================+=====================================================+
        |                                                       | The max value of the nth input activations into     |
        | ``activations/max/MODULE_NAME/input_{n}``             | the current module.                                 |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average value of the nth input activations      |
        | ``activations/average/MODULE_NAME/input_{n}``         | into the current module.                            |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The L2 Norm of the values of the nth input          |
        | ``activations/l2_norm/MODULE_NAME/input_{n}``         | activations into the current module.                |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The kurtosis of the values of the nth input         |
        | ``activations/kurtosis/MODULE_NAME/input_{n}``        | activations into the current module.                |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The max value of the nth ouput activations of the   |
        | ``activations/max/MODULE_NAME/output_{n}``            | current module.                                     |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The average value of the nth output activations of  |
        | ``activations/average/MODULE_NAME/output_{n}``        | the current module.                                 |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The L2 Norm of the values of nth output activations |
        | ``activations/l2_norm/MODULE_NAME/input_{n}``         | of the current module.                              |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+
        |                                                       | The kurtosis of the values of nth output            |
        | ``activations/kurtosis/MODULE_NAME/input_{n}``        | activations of the current module.                  |
        |                                                       |                                                     |
        +-------------------------------------------------------+-----------------------------------------------------+

    Args:
        interval (Union[int, str, Time], optional): Time string specifying how often to attach the logger and log the activations.
            For example, ``interval='5ba'`` means every 5 batches we log the activations. Default: '25ba'.
        ignore_module_types (Optional[List[str]], optional): A list of strings representing the class attributes we should ignore.
            For example passing in the list ['dropout', 'ln'] will cause the class attributes that contain
            'dropout' or 'ln' to not be logged. Default: 'None'.
        only_log_wandb (bool, optional): A bool that determines if we should only log to Weights and Biases. This is recommended
            in partcular for larger models as this callback logs a lot. Default: 'True'.
    """

    def __init__(self,
                 interval: Union[int, str, Time] = '25ba',
                 ignore_module_types: Optional[List[str]] = None,
                 only_log_wandb: bool = True):
        self.ignore_module_types = ignore_module_types
        self.only_log_wandb = only_log_wandb

        self.handles = []

        # Check that the interval timestring is parsable and convert into time object
        if isinstance(interval, int):
            self.interval = Time(interval, TimeUnit.BATCH)
        elif isinstance(interval, str):
            self.interval = Time.from_timestring(interval)
        elif isinstance(interval, Time):
            self.interval = interval

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
            self.recursively_add_metrics(metrics, module_name, f'_input.{i}', val)

        for i, val in enumerate(output):
            if val is None or isinstance(val, dict):
                continue
            self.recursively_add_metrics(metrics, module_name, f'_output.{i}', val)

        if self.only_log_wandb:
            wandb_loggers = [ld for ld in logger.destinations if isinstance(ld, WandBLogger)]
            if len(wandb_loggers):
                for wandb_logger in wandb_loggers:
                    wandb_logger.log_metrics(metrics, step)
            else:
                # In the case there were no WandB loggers, just default to
                # the standard logger and let it take care of it
                logger.log_metrics(metrics)
        else:
            logger.log_metrics(metrics)

    def recursively_add_metrics(self, metrics: dict, name: str, suffix: str, values: Any):
        # Keep recursively diving if the value is a sequence
        if isinstance(values, Sequence):
            for i, value in enumerate(values):
                self.recursively_add_metrics(metrics, f'{name}_{i}', suffix, value)
            return
        else:
            self.add_metrics(metrics, name, suffix, values)

    def add_metrics(self, metrics: dict, name: str, suffix: str, value: torch.Tensor):
        # We shouldn't log booleans
        if value.dtype == torch.bool:
            return
        if value.is_floating_point() or value.is_complex():
            metrics[f'activations/l2_norm/{name}{suffix}'] = torch.linalg.vector_norm(value).item()
            metrics[f'activations/average/{name}{suffix}'] = value.mean().item()
            metrics[f'activations/kurtosis/{name}{suffix}'] = self.compute_kurtosis(value).item()

        metrics[f'activations/max/{name}{suffix}'] = value.max().item()

    def compute_kurtosis(self, value: torch.Tensor):
        mean = value.mean()
        diffs = value - mean
        m_4 = torch.pow(diffs, 4).mean()
        var = torch.pow(diffs, 2).mean()
        return m_4 / (var**2)

    def create_module_names(self, model: torch.nn.Module):
        self.module_names = {m: name for name, m in model.named_modules()}
