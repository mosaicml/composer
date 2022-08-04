# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Outputs profiling data in JSON trace format."""

import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from composer.core.state import State
from composer.core.time import Timestamp
from composer.loggers import Logger
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.trace_handler import TraceHandler

__all__ = ['PerformanceAnalyzerTraceHandler']

MINIMUM_DATAPOINTS_NEEDED = 10


class PerformanceAnalyzerTraceHandler(TraceHandler):
    """Meaures if trainer run is model or dataloader bottlenecked."""

    def __init__(self):
        # Stores wait times for the current epoch
        self.dataloader_wait_times: Dict[int, Any] = {}
        self.batch_wait_times: Dict[int, Any] = {}
        self.last_start_time = None
        # Dataloader is bottlenecked if yield takes more than 0.1% of batch compute time
        self.DATALOADER_YIELD_THRESHOLD = 0.001

    def reset_wait_times(self, state: State, logger: Logger) -> None:
        """Reset wait times for new epoch as we might have changed algorithms in response to bottleneck.

        This function should be called to reset the trace handelr if we've made changes to the training
        loop which may change dataloader bottlenecked status.
        """
        self.dataloader_wait_times = {}
        self.batch_wait_times = {}

    def _record_wait_time(
        self,
        is_start: bool,
        wait_times: Dict[int, Any],
        wait_key: int,
        wall_clock_time_ns: int,
    ) -> None:
        if is_start:
            self.last_start_time = wall_clock_time_ns
        elif self.last_start_time:
            latency = wall_clock_time_ns - self.last_start_time
            wait_times[wait_key] = latency

    def process_duration_event(
        self,
        name: str,
        categories: Union[List[str], Tuple[str, ...]],
        is_start: bool,
        timestamp: Timestamp,
        wall_clock_time_ns: int,
        action: ProfilerAction,
    ) -> None:
        del name
        if 'dataloader' in categories:
            self._record_wait_time(is_start, self.dataloader_wait_times, timestamp.batch.value, wall_clock_time_ns)
        # Check for both [before/after]_train_batch and PerformanceAnalyzerTraceHandler so it only
        # fires once on [before/after]_train_batch
        elif 'before_train_batch' in categories and 'PerformanceAnalyzerTraceHandler' in categories:
            self._record_wait_time(True, self.batch_wait_times, timestamp.batch.value, wall_clock_time_ns)
        elif 'after_train_batch' in categories and 'PerformanceAnalyzerTraceHandler' in categories:
            self._record_wait_time(False, self.batch_wait_times, timestamp.batch.value, wall_clock_time_ns)
            # Only check for bottleneck on end of batch
            if not is_start and action == ProfilerAction.ACTIVE_AND_SAVE and self.is_dataloader_bottlenecked():
                warnings.warn(
                    f"The current training run is dataloader bottlenecked. Waiting {format(self._average_dataloader_wait_time() / 1e9, '.3f')}s per batch for dataloader."
                )

    def _average_batch_wait_time(self) -> float:
        return self._pruned_mean(list(self.batch_wait_times.values()))

    def _average_dataloader_wait_time(self) -> float:
        return self._pruned_mean(list(self.dataloader_wait_times.values()))

    def _pruned_mean(self, data: List[float], mstdev_threshold=2.0) -> float:
        """Compute pruned mean by filtering outliers using median standard deviations."""
        arr = np.asarray(data)
        deviations_from_median = np.abs(arr - np.median(arr))
        median_deviation = np.median(deviations_from_median)
        mstdevs = deviations_from_median / median_deviation if median_deviation else 0.
        filtered_arr = arr[mstdevs < mstdev_threshold]
        # Return mean of filtered_arr if it exists, otherwise return median of original array
        return float(np.mean(filtered_arr)) if len(filtered_arr) > 0 else np.median(arr)

    def is_dataloader_bottlenecked(self) -> bool:
        """Bottlenecked if batch yield time is more than 0.1% of batch compute time with >=10 datapoints."""
        return len(self.dataloader_wait_times) >= MINIMUM_DATAPOINTS_NEEDED and self._average_dataloader_wait_time(
        ) > self._average_batch_wait_time() * self.DATALOADER_YIELD_THRESHOLD
