# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""System metrics monitor callback."""

from __future__ import annotations

import logging
import os

import psutil
import torch

from composer.core import Callback, Event, State
from composer.loggers import Logger
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['SystemMetricsMonitor']

_GPU_METRICS = [
    'gpu_percentage',
    'memory_percentage',
    'gpu_temperature_C',
    'gpu_power_usage_W',
]


class SystemMetricsMonitor(Callback):
    """Logs GPU/CPU metrics.

    GPU Metrics:
        gpu_percentage: Occupancy rate, percent of time over sampling period during which one or more kernels was executing on the GPU.
        memory_percentage: Percent of time over sampling period during which global memory was being read or written.
        gpu_temperature_C: Temperature of device, in Celcius.
        gpu_power_usage_W: Power usage of device, in Watts.

    By default, only the maximum and minimum values for these metrics, alongside their respective ranks in the key names,
    are logged on the :attr:`.Event.BATCH_START`, :attr:`.Event.EVAL_BATCH_START`, :attr:`.Event.PREDICT_BATCH_START`
    events for every batch. If log_all_data is set to True, all values for these metrics across all ranks are logged on the
    above events for every batch.

    Example:
    .. doctest::

        >>> from composer import Trainer
        >>> from composer.callbacks import SystemMetricsMonitor
        >>> # constructing trainer object with this callback
        >>> trainer = Trainer(
        ...    model=model,
        ...    train_dataloader=train_dataloader,
        ...    eval_dataloader=eval_dataloader,
        ...    optimizers=optimizer,
        ...    max_duration='1ep',
        ...    callbacks=[SystemMetricsMonitor()],
        ... )

    Args:
        log_all_data (bool, optional): True if user wants to log data for all ranks, not just the min/max.
            Defaults to False.
    """

    def __init__(self, log_all_data: bool = False) -> None:
        super().__init__()
        self.gpu_available = torch.cuda.is_available()
        self.log_all_data = log_all_data
        if self.gpu_available:
            try:
                import pynvml
            except ImportError as e:
                raise MissingConditionalImportError(
                    extra_deps_group='pynvml',
                    conda_package='pynvml',
                    conda_channel='conda-forge',
                ) from e
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
            system_metrics = {}

            if self.log_all_data:
                for rank, metrics in enumerate(all_system_metrics):
                    for key, value in metrics.items():
                        if key in _GPU_METRICS:
                            system_metrics[f'{key}_rank_{rank}'] = value
                        else:
                            system_metrics[key] = value

            else:
                system_metrics = self.compute_gpu_min_max_metrics(all_system_metrics, state)
                for rank, metrics in enumerate(all_system_metrics):
                    for key, value in metrics.items():
                        if key not in _GPU_METRICS:
                            system_metrics[key] = value

            logger.log_metrics(system_metrics)

    def compute_system_metrics(self):
        system_metrics = {}

        # Get metrics for this device if available
        if self.gpu_available:
            import pynvml
            local_rank = dist.get_local_rank()
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            system_metrics['gpu_percentage'] = device_utilization.gpu
            system_metrics['memory_percentage'] = device_utilization.memory
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            system_metrics['gpu_temperature_C'] = temperature
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # convert from mW to W
            system_metrics['gpu_power_usage_W'] = power

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

    def compute_gpu_min_max_metrics(self, all_metrics, state):
        min_max_metrics = {}

        if self.gpu_available:
            for key in _GPU_METRICS:
                values = torch.tensor([metrics_for_cur_rank[key] for metrics_for_cur_rank in all_metrics])
                values = state.device.tensor_to_device(values)
                min_rank = int(torch.argmin(values).item())
                max_rank = int(torch.argmax(values).item())
                min_max_metrics[f'min_{key}_rank_{min_rank}'] = values[min_rank].item()
                min_max_metrics[f'max_{key}_rank_{max_rank}'] = values[max_rank].item()

        return min_max_metrics
