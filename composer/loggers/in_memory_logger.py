# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to dictionary objects that persist in memory throughout training.

Useful for collecting and plotting data inside notebooks.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from torch import Tensor

from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils.import_helpers import MissingConditionalImportError

if TYPE_CHECKING:
    from composer.core import State, Timestamp

__all__ = ['InMemoryLogger']


class InMemoryLogger(LoggerDestination):
    """Logs metrics to dictionary objects that persist in memory throughout training.

    Useful for collecting and plotting data inside notebooks.

    Example usage:
        .. testcode::

            from composer.loggers import InMemoryLogger
            from composer.trainer import Trainer
            logger = InMemoryLogger(
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

    Attributes:
        data (dict[str, list[tuple[Timestamp, Any]]]): Mapping of a logged key to
            a (:class:`~.time.Timestamp`, logged value) tuple.
            This dictionary contains all logged data.
        most_recent_values (dict[str, Any]): Mapping of a key to the most recent value for that key.
        most_recent_timestamps (dict[str, Timestamp]): Mapping of a key to the
            :class:`~.time.Timestamp` of the last logging call for that key.
        hyperparameters (dict[str, Any]): Dictionary of all hyperparameters.
        tables (dict[str, str]): dictionary of table name to json table.

    """

    def __init__(self) -> None:
        self.data: dict[str, list[tuple[Timestamp, Any]]] = {}
        self.most_recent_values = {}
        self.most_recent_timestamps: dict[str, Timestamp] = {}
        self.state: Optional[State] = None
        self.hyperparameters: dict[str, Any] = {}
        self.tables: dict[str, str] = {}

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        self.hyperparameters.update(hyperparameters)

    def log_table(
        self,
        columns: list[str],
        rows: list[list[Any]],
        name: str = 'Table',
        step: Optional[int] = None,
    ) -> None:
        del step
        try:
            import pandas as pd
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='pandas',
                conda_package='pandas',
                conda_channel='conda-forge',
            ) from e
        table = pd.DataFrame.from_records(data=rows,
                                          columns=columns).to_json(orient='split', index=False, force_ascii=False)
        assert table is not None
        # Merged assert is different
        # assert isinstance(table, str)
        self.tables[name] = table

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        assert self.state is not None
        timestamp = self.state.timestamp
        copied_metrics = copy.deepcopy(metrics)
        for k, v in copied_metrics.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((timestamp, v))
        self.most_recent_values.update(copied_metrics.items())
        self.most_recent_timestamps.update({k: timestamp for k in copied_metrics})

    def init(self, state: State, logger: Logger) -> None:
        self.state = state

    def get_timeseries(self, metric: str) -> dict[str, Any]:
        """Returns logged data as dict containing values of a desired metric over time.

        Args:
            metric (str): Metric of interest. Must be present in self.data.keys().

        Returns:
            timeseries (dict[str, Any]): Dictionary in which one key is ``metric``,
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

                from composer.loggers import InMemoryLogger
                from composer.core.time import Time, Timestamp

                in_mem_logger = InMemoryLogger()
                trainer = Trainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    max_duration="1ep",
                    optimizers=[optimizer],
                    loggers=[in_mem_logger]
                )

                # Populate the logger with data
                for b in range(0,3):
                    datapoint = b * 3
                    in_mem_logger.log_metrics({"accuracy/val": datapoint})

                timeseries = in_mem_logger.get_timeseries("accuracy/val")
                plt.plot(timeseries["batch"], timeseries["accuracy/val"])
                plt.xlabel("Batch")
                plt.ylabel("Validation Accuracy")
        """
        # Check that desired metric is in present data
        if metric not in self.data.keys():
            raise ValueError(
                f'Invalid value for argument `metric`: {metric}. Requested '
                'metric is not present in self.data.keys().',
            )

        timeseries = {}
        # Iterate through datapoints
        for datapoint in self.data[metric]:
            timestamp, metric_value = datapoint
            timeseries.setdefault(metric, []).append(metric_value)
            # Iterate through time units and add them all!
            for field, time in timestamp.state_dict().items():
                time_value = time if isinstance(time, int) else time.total_seconds()
                timeseries.setdefault(field, []).append(time_value)
        # Convert to numpy arrays
        for k, v in timeseries.items():
            if isinstance(v[0], Tensor):
                v = Tensor(v).numpy()
            else:
                v = np.array(v)
            timeseries[k] = v
        return timeseries
