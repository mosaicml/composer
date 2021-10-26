# Copyright 2021 MosaicML. All Rights Reserved.

from composer.core import Logger, State
from composer.core.callback import Callback
from torch.cuda import memory_stats, device_count


class MemoryMonitor(Callback):
    """Logs the memory usage of the model.
    

    It logs several memory usage statistics on each batch under the
    ``memory/{stastistic}`` key. If ``aggregate_device_stats`` is True, 
    (default False), then the statistics are computed across all GPUs.

    Args:
        aggregate_device_stats (bool, optional):
            Whether to compute memory statistics across all GPU devices.
    """

    def __init__(
        self,
        aggregate_device_stats: bool = False
    ):
        super().__init__()
        self.aggregate_device_stats = aggregate_device_stats

    def after_train_batch(self, state: State, logger: Logger):
        """
        This function calls the torch cuda memory stats and reports basic memory
        statistics. By default, it reports stats from GPU device 0, 
        but it can compute stats across all GPUs if the aggregate_device_stats
        parameter is true. To report additional statistics, add their
        torch names to default_stats.

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
            print("No CUDA devices found!")
            return

        if self.aggregate_device_stats:
            device_range = n_devices
        else:
            device_range = 1 # By default look at GPU 0 (should be representative)

        for d in range(device_range):
            device_stats = memory_stats()
            for torch_stat_name, stat_alias in default_stats.items():
                memory_report[stat_alias] = device_stats.get(torch_stat_name, 0) + \
                    memory_report.get(stat_alias, 0)

        for mem_stat, val in memory_report.items():
            logger.metric_batch({'memory/{}'.format(mem_stat): val})