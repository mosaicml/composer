# Copyright 2021 MosaicML. All Rights Reserved.

"""Log memory usage during training."""
import logging
from typing import Dict, Union

import torch.cuda

from composer.core import Logger, State
from composer.core.callback import Callback

log = logging.getLogger(__name__)

__all__ = ["MemoryMonitor"]


class MemoryMonitor(Callback):
    """Logs the memory usage of the model.

    This callback calls the torch memory stats API for cuda (see :func:`torch.cuda.memory_stats`) on the
    :attr:`~composer.core.event.Event.AFTER_TRAIN_BATCH` and reports different memory statistics.

    Example
       >>> # constructing trainer object with this callback
       >>> trainer = Trainer(
       ...     model=model,
       ...     train_dataloader=train_dataloader,
       ...     eval_dataloader=eval_dataloader,
       ...     optimizers=optimizer,
       ...     max_duration="1ep",
       ...     callbacks=[callbacks.MemoryMonitor()],
       ... )

    The memory statistics are logged by the :class:`~composer.core.logging.logger.Logger` to the following keys as
    described below.

    +--------------------------+-------------------------------------------------------------+
    | Key                      | Logged data                                                 |
    +==========================+=============================================================+
    |                          | Several memory usage statistics                             |
    | ``memory/{statistic}``   | are logged on                                               |
    |                          | :attr:`~composer.core.event.Event.AFTER_TRAIN_BATCH` event. |
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
        Memory usage monitoring is only supported for the GPU devices.
    """

    def __init__(self):
        super().__init__()
        log.info(
            "Memory monitor just profiles the current GPU assuming that the memory footprint across GPUs is balanced.")
        if torch.cuda.device_count() == 0:
            log.warn("Memory monitor only works on GPU devices.")

    def after_train_batch(self, state: State, logger: Logger):
        memory_report = {}

        n_devices = torch.cuda.device_count()
        if n_devices == 0:
            return

        memory_report = _get_memory_report()

        for mem_stat, val in memory_report.items():
            logger.metric_batch({'memory/{}'.format(mem_stat): val})


_MEMORY_STATS = {
    "allocation.all.allocated": "alloc_requests",
    "allocation.all.freed": "free_requests",
    "allocated_bytes.all.allocated": "allocated_mem",
    "active_bytes.all.current": "active_mem",
    "inactive_split_bytes.all.current": "inactive_mem",
    "reserved_bytes.all.current": "reserved_mem",
    "num_alloc_retries": "alloc_retries",
}


def _get_memory_report() -> Dict[str, Union[int, float]]:
    if not torch.cuda.is_available():
        log.debug("Cuda is not available. The memory report will be empty.")
        return {}
    memory_stats = torch.cuda.memory_stats()

    if len(memory_stats) == 0:
        log.debug("No GPU memory was used, returning empty.")
        return {}

    # simplify the memory_stats
    memory_report = {
        name: memory_stats[torch_name] for (torch_name, name) in _MEMORY_STATS.items() if torch_name in memory_stats
    }

    return memory_report
