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
        self.dataloader_wait_times: Dict[int, float] = {}
        self.batch_wait_times: Dict[int, float] = {}
        self.last_start_time = None
        # Dataloader is bottlenecked if yield takes more than 1% of batch compute time and at least
        # 1 millisecond. We use a percentage threshold as the primary metric users care about, but
        # we also need to put a floor millisecond threshold to account for python overhead in
        # yielding a batch, which could potentially exceed 1% for very small models.
        self.DATALOADER_PERCENTAGE_THRESHOLD = 0.01
        self.DATALOADER_TIME_THRESHOLD = 1.0e6  # 1 millisecond or 1e6 nanoseconds

    def reset_wait_times(self, state: State, logger: Logger) -> None:
        """Reset wait times if algorithms have changed in response to bottleneck.

        This function should be called to reset the trace handler if any changes have been made to the
        training loop which may change dataloader bottlenecked status.
        """
        del state, logger
        self.dataloader_wait_times = {}
        self.batch_wait_times = {}

    def epoch_start(self, state: State, logger: Logger) -> None:
        self.reset_wait_times(state, logger)

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
                dataloader_time = self._average_dataloader_wait_time()
                batch_time = self._average_batch_wait_time()
                # Log percentage overhead and additional time in milliseconds
                percentage = dataloader_time / batch_time * 100
                additional_latency = dataloader_time / 1e6
                warnings.warn(
                    f'The current training run is dataloader bottlenecked, adding {additional_latency:.3f}ms and '
                    f'increasing batch time by {percentage:.2f}%. This warning may be spurious or noisy if the '
                    '`active` period of the profiler schedule is too small.')

    def _average_batch_wait_time(self) -> float:
        return self._pruned_mean(list(self.batch_wait_times.values()))

    def _average_dataloader_wait_time(self) -> float:
        return self._pruned_mean(list(self.dataloader_wait_times.values()))

    def _pruned_mean(self, data: List[float], mstdev_threshold: float = 2.0) -> float:
        """Compute pruned mean by filtering outliers using median standard deviations."""
        arr = np.asarray(data)
        deviations_from_median = np.abs(arr - np.median(arr))
        median_deviation = np.median(deviations_from_median)
        mstdevs = deviations_from_median / median_deviation if median_deviation else 0.
        filtered_arr = arr[mstdevs < mstdev_threshold]
        # Return mean of filtered_arr if it exists, otherwise return median of original array
        return float(np.mean(filtered_arr)) if len(filtered_arr) > 0 else float(np.median(arr))

    def is_dataloader_bottlenecked(self) -> bool:
        """Bottlenecked if batch yield time is more than 0.1% of batch compute time with >=10 datapoints."""
        dataloader_time = self._average_dataloader_wait_time()
        batch_time = self._average_batch_wait_time()
        return len(
            self.dataloader_wait_times
        ) >= MINIMUM_DATAPOINTS_NEEDED and dataloader_time > batch_time * self.DATALOADER_PERCENTAGE_THRESHOLD and dataloader_time > self.DATALOADER_TIME_THRESHOLD
