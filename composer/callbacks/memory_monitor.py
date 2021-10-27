# Copyright 2021 MosaicML. All Rights Reserved.

import logging

from torch.cuda import device_count, memory_stats

from composer.core import Logger, State
from composer.core.callback import Callback

log = logging.getLogger(__name__)


class MemoryMonitor(Callback):
    """Logs the memory usage of the model.

    Logs several memory usage statistics on each batch under
    the ``memory/{statistic}`` key.

    Args:
    """

    def __init__(self):
        super().__init__()
        log.info(
            "Memory monitor just profiles the current GPU assuming that the memory footprint across GPUs is balanced.")
        if device_count == 0:
            log.warn("Memory monitor only works on GPU devices.")

    def after_train_batch(self, state: State, logger: Logger):
        """This function calls the torch cuda memory stats and reports basic memory
        statistics.

        Args:
            state (State): The :class:`~composer.core.State` object
                used during training.
            logger (Logger):
                The :class:`~composer.core.logging.logger.Logger` object.
        """
        memory_report = {}

        default_stats = {
            "allocation.all.allocated": "alloc_requests",
            "allocation.all.freed": "free_requests",
            "allocated_bytes.all.allocated": "allocated_mem",
            "active_bytes.all.current": "active_mem",
            "inactive_split_bytes.all.current": "inactive_mem",
            "reserved_bytes.all.current": "reserved_mem",
            "num_alloc_retries": "alloc_retries",
        }

        n_devices = device_count()
        if n_devices == 0:
            return

        device_stats = memory_stats()
        for torch_stat_name, stat_alias in default_stats.items():
            memory_report[stat_alias] = device_stats.get(torch_stat_name, 0)

        for mem_stat, val in memory_report.items():
            logger.metric_batch({'memory/{}'.format(mem_stat): val})
