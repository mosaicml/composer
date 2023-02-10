# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory usage during training."""
import logging
import math
import warnings
from typing import Dict, Optional, Union

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

    +----------------+-----------------------------------------------------------------------------------+
    | Statistic      | Description                                                                       |
    +================+===================================================================================+
    | allocated_mem  | Amount of allocated memory in gigabytes.                                          |
    +----------------+-----------------------------------------------------------------------------------+
    | active_mem     | Amount of active memory in gigabytes at the time of recording.                    |
    +----------------+-----------------------------------------------------------------------------------+
    | inactive_mem   | Amount of inactive, non-releaseable memory in gigabytes at the time of recording. |
    +----------------+-----------------------------------------------------------------------------------+
    | reserved_mem   | Amount of reserved memory in gigabytes at the time of recording.                  |
    +----------------+-----------------------------------------------------------------------------------+
    | alloc_retries  | Number of failed cudaMalloc calls that result in a cache flush and retry.         |
    +----------------+-----------------------------------------------------------------------------------+

    .. note::
        Memory usage monitoring is only supported for GPU devices.

    Args:
        memory_keys (Dict[str, str], optional): A dict specifying memory statistics to log. Keys
            are the names of memory statistics to log from `torch.cuda.memory_stats()`, and values
            are the names they will be logged under. If not provided, the above statistics are
            logged. Defaults to None.
    """

    def __init__(self, memory_keys: Optional[Dict[str, str]] = None) -> None:
        self.memory_keys = memory_keys

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

        memory_report = _get_memory_report(self.memory_keys)

        logger.log_metrics({f'memory/{mem_stat}': val for (mem_stat, val) in memory_report.items()})


_MEMORY_KEYS = {
    'allocated_bytes.all.current': 'allocated_mem',
    'active_bytes.all.current': 'active_mem',
    'inactive_split_bytes.all.current': 'inactive_mem',
    'reserved_bytes.all.current': 'reserved_mem',
    'num_alloc_retries': 'alloc_retries',
}


def _get_memory_report(memory_keys: Optional[Dict[str, str]] = None) -> Dict[str, Union[int, float]]:
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
