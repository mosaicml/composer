# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory usage during training."""
import logging
import warnings
from typing import Dict, Union

import torch.cuda

from composer.core import Callback, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['MemoryMonitor']


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

    +----------------+--------------------------------------------------------------------------------+
    | Statistic      | Description                                                                    |
    +================+================================================================================+
    | alloc_requests | Number of memory allocation requests received by the memory allocator.         |
    +----------------+--------------------------------------------------------------------------------+
    | free_requests  | Number of memory free requests received by the memory allocator.               |
    +----------------+--------------------------------------------------------------------------------+
    | allocated_mem  | Amount of allocated memory in bytes.                                           |
    +----------------+--------------------------------------------------------------------------------+
    | active_mem     | Amount of active memory in bytes at the time of recording.                     |
    +----------------+--------------------------------------------------------------------------------+
    | inactive_mem   | Amount of inactive, non-releaseable memory in bytes at the time of recording.  |
    +----------------+--------------------------------------------------------------------------------+
    | reserved_mem   | Amount of reserved memory in bytes at the time of recording.                   |
    +----------------+--------------------------------------------------------------------------------+
    | alloc_retries  | Number of failed cudaMalloc calls that result in a cache flush and retry.      |
    +----------------+--------------------------------------------------------------------------------+

    .. note::
        Memory usage monitoring is only supported for GPU devices.
    """

    def __init__(self) -> None:
        # Memory monitor takes no args
        pass

    def init(self, state: State, logger: Logger) -> None:
        # Not relying on `torch.cuda.is_available()` since the model could be on CPU.
        model_device = next(state.model.parameters()).device

        if model_device.type != 'cuda':
            warnings.warn(f'The memory monitor only works on CUDA devices, but the model is on {model_device.type}.')

    def after_train_batch(self, state: State, logger: Logger):
        memory_report = {}

        model_device = next(state.model.parameters()).device
        if model_device.type != 'cuda':
            return

        memory_report = _get_memory_report()

        logger.log_metrics({f'memory/{mem_stat}': val for (mem_stat, val) in memory_report.items()})


_MEMORY_STATS = {
    'allocation.all.allocated': 'alloc_requests',
    'allocation.all.freed': 'free_requests',
    'allocated_bytes.all.allocated': 'allocated_mem',
    'active_bytes.all.current': 'active_mem',
    'inactive_split_bytes.all.current': 'inactive_mem',
    'reserved_bytes.all.current': 'reserved_mem',
    'num_alloc_retries': 'alloc_retries',
}


def _get_memory_report() -> Dict[str, Union[int, float]]:
    memory_stats = torch.cuda.memory_stats()

    # simplify the memory_stats
    memory_report = {
        name: memory_stats[torch_name] for (torch_name, name) in _MEMORY_STATS.items() if torch_name in memory_stats
    }

    return memory_report
