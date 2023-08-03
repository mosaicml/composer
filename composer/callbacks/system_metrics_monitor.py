# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""System metrics monitor callback."""

from __future__ import annotations

import logging

import psutil
from py3nvml import py3nvml
import os

from composer.core import Callback, State, Event
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ["SystemMetricsMonitor"]


class SystemMetricsMonitor(Callback):
    """Track system metrics."""

    def run_event(self, event: Event, state: State, logger: Logger):
        # run on every event
        system_metrics = self.compute_system_metrics()
        logger.log_metrics(system_metrics)

    def compute_system_metrics(self):
        deviceCount = py3nvml.nvmlDeviceGetCount()
        system_metrics = {}

        # Get metrics for each device
        for i in range(0, deviceCount):
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
            memory = py3nvml.nvmlDeviceGetMemoryInfo(handle)
            system_metrics[f"device{i}_memory_total"] = memory.total
            system_metrics[f"device{i}_memory_free"] = memory.free
            system_metrics[f"device{i}_memory_used"] = memory.used
            device_utilization = py3nvml.nvmlDeviceGetUtilizationRates(handle)
            system_metrics[f"device{i}_gpu_percentage"] = device_utilization.gpu
            system_metrics[f"device{i}_memory_percentage"] = device_utilization.memory
            temperature = py3nvml.nvmlDeviceGetTemperature(
                handle, py3nvml.NVML_TEMPERATURE_GPU
            )
            system_metrics[f"device{i}_gpu_temperature"] = temperature

        # Get metrics for the system
        cpu_percent = psutil.cpu_percent()
        system_metrics[f"cpu_percentage"] = cpu_percent
        system_memory = psutil.virtual_memory()._asdict()
        for k, v in system_memory.items():
            system_metrics[f"cpu_memory_{k}"] = v
        disk_usage = psutil.disk_usage(os.sep)._asdict()
        for k, v in disk_usage.items():
            system_metrics[f"disk_memory_{k}"] = v
        network_usage = psutil.net_io_counters()._asdict()
        for k, v in network_usage.items():
            system_metrics[f"network_{k}"] = v
        return system_metrics
