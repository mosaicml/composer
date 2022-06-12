# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to dictionary objects that persist in memory throughout training.

Useful for collecting and plotting data inside notebooks.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from torch import Tensor

from composer.core.state import State
from composer.core.time import Time, Timestamp
from composer.loggers.logger import LogLevel
from composer.loggers.logger_destination import LoggerDestination

__all__ = ['InMemoryLogger']


class InMemoryLogger(LoggerDestination):
    """Logs metrics to dictionary objects that persist in memory throughout training.

    Useful for collecting and plotting data inside notebooks.

    Example usage:
        .. testcode::

            from composer.loggers import InMemoryLogger, LogLevel
            from composer.trainer import Trainer
            logger = InMemoryLogger(
                log_level=LogLevel.BATCH
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                optimizers=[optimizer],
                loggers=[logger]
            )
            # Get data from logger. If you are using multiple loggers, be sure to confirm
            # which index in trainer.logger.destinations contains your desired logger.
            logged_data = trainer.logger.destinations[0].data

    Args:
        log_level (str | LogLevel, optional):
            :class:`~.logger.LogLevel` (i.e. unit of resolution) at
            which to record. Defaults to
            :attr:`~.LogLevel.BATCH`, which records
            everything.

    Attributes:
        data (Dict[str, List[Tuple[Timestamp, LogLevel, Any]]]): Mapping of a logged key to
            a (:class:`~.time.Timestamp`, :class:`~.logger.LogLevel`, logged value) tuple.
            This dictionary contains all logged data.
        most_recent_values (Dict[str, Any]): Mapping of a key to the most recent value for that key.
        most_recent_timestamps (Dict[str, Timestamp]): Mapping of a key to the
            :class:`~.time.Timestamp` of the last logging call for that key.
    """

    def __init__(self, log_level: Union[str, int, LogLevel] = LogLevel.BATCH) -> None:
        self.log_level = LogLevel(log_level)
        self.data: Dict[str, List[Tuple[Timestamp, LogLevel, Any]]] = {}
        self.most_recent_values = {}
        self.most_recent_timestamps: Dict[str, Timestamp] = {}

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        if log_level > self.log_level:
            # the logged metric is more verbose than what we want to record.
            return
        timestamp = state.timestamp
        copied_data = copy.deepcopy(data)
        for k, v in copied_data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((timestamp, log_level, v))
        self.most_recent_values.update(copied_data.items())
        self.most_recent_timestamps.update({k: timestamp for k in copied_data})

    def get_timeseries(self, metric: str) -> Dict[str, Any]:
        """Returns logged data as dict containing values of a desired metric over time.

        Args:
            metric (str): Metric of interest. Must be present in self.data.keys().

        Returns:
            timeseries (Dict[str, Any]): Dictionary in which one key is ``metric``,
                and the associated value is a list of values of that metric. The remaining
                keys are each a unit of time, and the associated values are each a list of
                values of that time unit for the corresponding index of the metric. For
                example:
                >>> InMemoryLogger.get_timeseries(metric="accuracy/val")
                {"accuracy/val": [31.2, 45.6, 59.3, 64.7, "epoch": [1, 2, 3, 4, ...],
                ...], "batch": [49, 98, 147, 196, ...], ...}

        Example:
            .. testcode::

                import matplotlib.pyplot as plt

                from composer.loggers import InMemoryLogger, LogLevel
                from composer.core.time import Time, Timestamp

                in_mem_logger = InMemoryLogger(LogLevel.BATCH)

                # Populate the logger with data
                for b in range(0,3):
                    datapoint = b * 3
                    in_mem_logger.log_data(state=state, log_level=LogLevel.BATCH, data={"accuracy/val": datapoint})

                timeseries = in_mem_logger.get_timeseries("accuracy/val")
                plt.plot(timeseries["batch"], timeseries["accuracy/val"])
                plt.xlabel("Batch")
                plt.ylabel("Validation Accuracy")
        """
        # Check that desired metric is in present data
        if metric not in self.data.keys():
            raise ValueError(f'Invalid value for argument `metric`: {metric}. Requested '
                             'metric is not present in self.data.keys().')

        timeseries = {}
        # Iterate through datapoints
        for datapoint in self.data[metric]:
            timestamp, _, metric_value = datapoint
            timeseries.setdefault(metric, []).append(metric_value)
            # Iterate through time units and add them all!
            for field, time in timestamp.get_state().items():
                time_value = time.value if isinstance(time, Time) else time.total_seconds()
                timeseries.setdefault(field, []).append(time_value)
        # Convert to numpy arrays
        for k, v in timeseries.items():
            if isinstance(v[0], Tensor):
                v = Tensor(v).numpy()
            else:
                v = np.array(v)
            timeseries[k] = v
        return timeseries
