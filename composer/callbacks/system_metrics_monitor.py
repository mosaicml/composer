# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""System metrics monitor callback."""

from __future__ import annotations

import logging
import os
from typing import Union

import torch
import psutil

from composer.core import Callback, Event, State
from composer.loggers import Logger
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

log = logging.getLogger(__name__)

__all__ = ['SystemMetricsMonitor']


class SystemMetricsMonitor(Callback):
    """Logs the minimum and maximum training values across all ranks for the following metrics:

        RoundTripTime: Time spent in all the traced ops in the current batch
        Power: GPU Power Consumption
        Temp: GPU Temperature
        Utilization: GPU Utilization
        Clock: GPU Clock
        BatchLoadLatency: Time spent loading the current batch from the dataset
        Throughput: Estimated throughput for the current batch

    The maximum and minimum values for these metrics, alongside their respective ranks, are logged 
    on the :attr:`.Event.BATCH_END` event for every batch. 

    To compute `flops_per_sec`, the model attribute `flops_per_batch` should be set to a callable
    which accepts a batch and returns the number of flops for that batch. Typically, this should
    be flops per sample times the batch size unless pad tokens are used.

    The wall clock time is logged on every :attr:`.Event.BATCH_END` event.

    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import GlobalStragglerDetector
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration='1ep',
            ...     callbacks=[GlobalStragglerDetector()],
            ... )

    The metrics are logged by the :class:`.Logger` to the following keys as
    described below.

    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Minimum time spent in all the traced ops in the           |
    | `MinRoundTripTime/Rank`             | current batch across all ranks for the corresponding rank |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Maximum time spent in all the traced ops in the           |
    | `MaxRoundTripTime/Rank`             | current batch across all ranks for the corresponding rank |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinPower/Rank`                     | Minimum GPU Power consumed for the corresponding rank     |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxPower/Rank`                     | Maximum GPU Power consumed for the corresponding rank     |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinTemp/Rank`                      | Minimum GPU Temperature for the corresponding rank        |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxTemp/Rank`                      | Maximum GPU Temperature for the corresponding rank        |    
    +-------------------------------------+-----------------------------------------------------------+
    | `MinUtilization/Rank`               | Minimum GPU Utilization for the corresponding rank        |
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxUtilization/Rank`               | Maximum GPU Utilization for the corresponding rank        |  
    +-------------------------------------+-----------------------------------------------------------+
    | `MinClock/Rank`                     | Minimum GPU Clock for the corresponding rank              |  
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxClock/Rank`                     | Maximum GPU Clock for the corresponding rank              |  
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Minimum time spent loading the current batch from the     |
    | `MinBatchLoadLatency/Rank`          | dataset across all ranks for the corresponding rank       |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Maximum time spent loading the current batch from the     |
    | `MaxBatchLoadLatency/Rank`          | dataset across all ranks for the corresponding rank       |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    | `MinThroughput/Rank`                | Minimum estimated throughput for the corresponding rank   |  
    +-------------------------------------+-----------------------------------------------------------+
    | `MaxThroughput/Rank`                | Maximum estimated throughput for the corresponding rank   |  
    +-------------------------------------+-----------------------------------------------------------+

    
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
            if self.log_all_data:
                system_metrics = {
                    key: value for local_metrics in all_system_metrics for key, value in local_metrics.items()
                }
            else:
                system_metrics = self.compute_min_max_metrics(all_system_metrics)
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
            system_metrics['rank'] = global_rank
            system_metrics['memory_total_bytes'] = memory.total
            system_metrics['memory_free_bytes'] = memory.free
            system_metrics['memory_used_bytes'] = memory.used
            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            system_metrics['gpu_percentage'] = device_utilization.gpu
            system_metrics['memory_percentage'] = device_utilization.memory
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            system_metrics['gpu_temperature_C'] = temperature
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # convert from mW to W
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


    def compute_min_max_metrics(self, all_metrics):
        min_metrics = {}
        max_metrics = {}
        min_max_metrics = {}

        for key, value in all_metrics[0].items():
            if key.startswith('device'):
                metric_name = key.split('_', 1)[1]
                min_metrics[metric_name] = (value, 0)  
                max_metrics[metric_name] = (value, 0)
            else:
                min_max_metrics[key] = value
        
      
        for cur_rank, metrics in enumerate(all_metrics[1:]):
            for key, value in metrics.items():
                if key.startswith('device'):
                    metric_name = key.split('_', 1)[1]
                    current_min_value, _ = min_metrics[metric_name]
                    current_max_value, _ = max_metrics[metric_name]
                    if value < current_min_value:
                        min_metrics[metric_name] = (value, cur_rank)
                    elif value > current_max_value:
                        max_metrics[metric_name] = (value, cur_rank)
                else:
                    min_max_metrics[key] = value

        for key, _ in min_metrics.items():
            min_rank = min_metrics[key][1]
            max_rank = max_metrics[key][1]
            min_max_metrics["min_" + key + "/Rank_" + str(min_rank)] = min_metrics[key][0]
            min_max_metrics["max_" + key + "/Rank_" + str(max_rank)] = max_metrics[key][0]

        return min_max_metrics