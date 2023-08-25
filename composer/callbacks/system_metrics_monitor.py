# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""System metrics monitor callback."""

from __future__ import annotations

import logging
import os

import psutil

from composer.core import Callback, Event, State
from composer.loggers import Logger
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['SystemMetricsMonitor']


class SystemMetricsMonitor(Callback):
    """Track system metrics."""

    def __init__(self, gpu_available: bool = False) -> None:
        super().__init__()
        self.gpu_available = gpu_available
        if self.gpu_available:
            try:
                import pynvml
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='pynvml',
                                                    conda_package='pynvml',
                                                    conda_channel='conda-forge') from e
            pynvml.nvmlInit()

    def run_event(self, event: Event, state: State, logger: Logger):
        # only run on the following events
        if event in [
                Event.BATCH_START,
                Event.EVAL_BATCH_START,
                Event.PREDICT_BATCH_START,
        ]:
            local_node_system_metrics = self.compute_system_metrics()
            all_system_metrics = dist.all_gather_object(local_node_system_metrics)
            system_metrics = {
                key: value for local_metrics in all_system_metrics for key, value in local_metrics.items()
            }
            logger.log_metrics(system_metrics)

    def compute_system_metrics(self):
        system_metrics = {}

        # Get metrics for this device if available
        if self.gpu_available:
            import pynvml
            local_rank = dist.get_local_rank()
            global_rank = dist.get_global_rank()
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            system_metrics[f'device{global_rank}_memory_total'] = memory.total
            system_metrics[f'device{global_rank}_memory_free'] = memory.free
            system_metrics[f'device{global_rank}_memory_used'] = memory.used
            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            system_metrics[f'device{global_rank}_gpu_percentage'] = device_utilization.gpu
            system_metrics[f'device{global_rank}_memory_percentage'] = device_utilization.memory
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            system_metrics[f'device{global_rank}_gpu_temperature'] = temperature

        # Get metrics for the system
        cpu_percent = psutil.cpu_percent()
        system_metrics[f'cpu_percentage'] = cpu_percent
        system_memory = psutil.virtual_memory()._asdict()
        for k, v in system_memory.items():
            system_metrics[f'cpu_memory_{k}'] = v
        disk_usage = psutil.disk_usage(os.sep)._asdict()
        for k, v in disk_usage.items():
            system_metrics[f'disk_memory_{k}'] = v
        network_usage = psutil.net_io_counters()._asdict()
        for k, v in network_usage.items():
            system_metrics[f'network_{k}'] = v
        return system_metrics
