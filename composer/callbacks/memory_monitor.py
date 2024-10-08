# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory usage during training."""
import logging
import math
import warnings
from typing import Optional, Union

import torch
import torch.cuda
from torch import distributed

from composer.core import Callback, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['MemoryMonitor']


def reduce_value(
    value: Union[int, float],
    model_device: torch.device,
    reduce_op: str = 'mean',
):
    """Reduce a value across distributed processes.

    Args:
        value (Union[int, float]): The value to reduce.
        model_device (torch.device): The device on which the model is located.
        reduce_op (str, optional): The reduction operation to perform. One of 'mean', 'avg', 'sum', 'min', 'max'.
            Defaults to 'mean'.
    """
    tensor_value = torch.tensor(value, device=model_device)

    if reduce_op in ['mean', 'avg', 'sum']:
        op = distributed.ReduceOp.SUM
    elif reduce_op == 'min':
        op = distributed.ReduceOp.MIN
    elif reduce_op == 'max':
        op = distributed.ReduceOp.MAX
    else:
        raise ValueError(f'{reduce_op=} not supported.')

    distributed.all_reduce(tensor_value, op=op)
    if reduce_op in ['mean', 'avg']:
        tensor_value = tensor_value / distributed.get_world_size()

    return tensor_value.item()


class MemoryMonitor(Callback):
    """Logs the memory usage of the model.

    This callback calls the torch memory stats API for CUDA (see :func:`torch.cuda.memory_stats`)
    on the :attr:`.Event.AFTER_TRAIN_BATCH` and reports different memory statistics.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import MemoryMonitor
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[MemoryMonitor()],
            ... )

    The memory statistics are logged by the :class:`.Logger` to the following keys as
    described below.

    +--------------------------+-------------------------------------------------------------+
    | Key                      | Logged data                                                 |
    +==========================+=============================================================+
    |                          | Several memory usage statistics                             |
    | ``memory/{statistic}``   | are logged on                                               |
    |                          | :attr:`.Event.AFTER_TRAIN_BATCH` event.                     |
    +--------------------------+-------------------------------------------------------------+

    The following statistics are recorded:

    +------------------------+-------------------------------------------------------------------------------------------+
    | Statistic              | Description                                                                               |
    +========================+===========================================================================================+
    | current_allocated_mem  | Current amount of allocated memory in gigabytes.                                          |
    +------------------------+-------------------------------------------------------------------------------------------+
    | current_active_mem     | Current amount of active memory in gigabytes at the time of recording.                    |
    +------------------------+-------------------------------------------------------------------------------------------+
    | current_inactive_mem   | Current amount of inactive, non-releaseable memory in gigabytes at the time of recording. |
    +------------------------+-------------------------------------------------------------------------------------------+
    | current_reserved_mem   | Current amount of reserved memory in gigabytes at the time of recording.                  |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_allocated_mem     | Peak amount of allocated memory in gigabytes.                                             |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_active_mem        | Peak amount of active memory in gigabytes at the time of recording.                       |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_inactive_mem      | Peak amount of inactive, non-releaseable memory in gigabytes at the time of recording.    |
    +------------------------+-------------------------------------------------------------------------------------------+
    | peak_reserved_mem      | Peak amount of reserved memory in gigabytes at the time of recording.                     |
    +------------------------+-------------------------------------------------------------------------------------------+
    | alloc_retries          | Number of failed cudaMalloc calls that result in a cache flush and retry.                 |
    +------------------------+-------------------------------------------------------------------------------------------+

    Additionally, if `dist_aggregate_batch_interval` is enabled, the `avg`, `min`, and `max` of the
    aformentioned statistics are also logged.

    .. note::
        Memory usage monitoring is only supported for GPU devices.

    Args:
        memory_keys (dict[str, str], optional): A dict specifying memory statistics to log. Keys
            are the names of memory statistics to log from `torch.cuda.memory_stats()`, and values
            are the names they will be logged under. If not provided, the above statistics are
            logged. Defaults to None.
        dist_aggregate_batch_interval (int, optional): interval for aggregating memory stats across
            all nodes. Defaults to None (by default the functionality is disabled).
    """

    def __init__(
        self,
        memory_keys: Optional[dict[str, str]] = None,
        dist_aggregate_batch_interval: Optional[int] = None,
    ) -> None:
        self.memory_keys = memory_keys
        self.dist_aggregate_batch_interval = dist_aggregate_batch_interval

    def init(self, state: State, logger: Logger) -> None:
        # Not relying on `torch.cuda.is_available()` since the model could be on CPU.
        model_device = next(state.model.parameters()).device

        if model_device.type not in ('cuda', 'meta'):
            warnings.warn(f'The memory monitor only works on CUDA devices, but the model is on {model_device.type}.')

    def after_train_batch(self, state: State, logger: Logger):
        memory_report = {}

        model_device = next(state.model.parameters()).device
        if model_device.type != 'cuda':
            return

        memory_report = _get_memory_report(self.memory_keys)
        if self.dist_aggregate_batch_interval is not None and state.timestamp.batch.value % self.dist_aggregate_batch_interval == 0:
            dist_memory_report = {}
            for (mem_stat, val) in memory_report.items():
                dist_memory_report[mem_stat + '_avg'] = reduce_value(val, model_device, 'avg')
                dist_memory_report[mem_stat + '_min'] = reduce_value(val, model_device, 'min')
                dist_memory_report[mem_stat + '_max'] = reduce_value(val, model_device, 'max')
            memory_report.update(dist_memory_report)

        logger.log_metrics({f'memory/{mem_stat}': val for (mem_stat, val) in memory_report.items()})


_MEMORY_KEYS = {
    'allocated_bytes.all.current': 'current_allocated_mem',
    'active_bytes.all.current': 'current_active_mem',
    'inactive_split_bytes.all.current': 'current_inactive_mem',
    'reserved_bytes.all.current': 'current_reserved_mem',
    'allocated_bytes.all.peak': 'peak_allocated_mem',
    'active_bytes.all.peak': 'peak_active_mem',
    'inactive_split_bytes.all.peak': 'peak_inactive_mem',
    'reserved_bytes.all.peak': 'peak_reserved_mem',
    'num_alloc_retries': 'alloc_retries',
}


def _get_memory_report(memory_keys: Optional[dict[str, str]] = None) -> dict[str, Union[int, float]]:
    memory_stats = torch.cuda.memory_stats()
    memory_keys = memory_keys or _MEMORY_KEYS

    # simplify and reformat the memory_stats
    memory_report = {}
    for (torch_name, name) in memory_keys.items():
        if torch_name in memory_stats:
            # Convert to gigabytes
            if 'bytes' in torch_name:
                gigabytes = memory_stats[torch_name] / 1.0e9
                # Round to preserve 5 significant digits
                if gigabytes != 0:
                    order_of_magnitude = int(math.floor(math.log10(abs(gigabytes))))
                    gigabytes = round(gigabytes, -order_of_magnitude + 4)
                memory_report[name.replace('bytes', 'gigabytes')] = gigabytes
            else:
                memory_report[name] = memory_stats[torch_name]

    return memory_report
