# Copyright 2021 MosaicML. All Rights Reserved.

"""Logs metrics to dictionary objects that persist in memory throughout training.

Useful for collecting and plotting data inside notebooks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

from composer.core.logging import LoggerCallback, LogLevel, TLogData
from composer.core.logging.logger import TLogDataValue
from composer.core.time import Timestamp

__all__ = ["InMemoryLogger"]


class InMemoryLogger(LoggerCallback):
    """Logs metrics to dictionary objects that persist in memory throughout training. Useful for collecting and plotting
    data inside notebooks.

    Example usage:
        .. testcode::

            from composer.loggers import InMemoryLogger
            from composer.trainer import Trainer
            from composer.core.logging import LogLevel
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
            # which index in trainer.logger.backends contains your desired logger.
            logged_data = trainer.logger.backends[0].data

    Args:
        log_level (str or LogLevel, optional):
            :class:`~.logger.LogLevel` (i.e. unit of resolution) at
            which to record. Defaults to
            :attr:`~.LogLevel.BATCH`, which records
            everything.

    Attributes:
        data (dict): Mapping of a logged key to a
            (:class:`~.time.Timestamp`, :class:`~.logger.LogLevel`,
            :attr:`~.logger.TLogDataValue`) tuple. This dictionary contains all logged
            data.
        most_recent_values (Dict[str, TLogData]): Mapping of a key to the most recent value for that key.
        most_recent_timestamps (Dict[str, Timestamp]): Mapping of a key to the
            :class:`~.time.Timestamp` of the last logging call for that key.
    """

    def __init__(self, log_level: Union[str, LogLevel] = LogLevel.BATCH) -> None:
        self.log_level = LogLevel(log_level)
        self.data: Dict[str, List[Tuple[Timestamp, LogLevel, TLogDataValue]]] = {}
        self.most_recent_values: Dict[str, TLogDataValue] = {}
        self.most_recent_timestamps: Dict[str, Timestamp] = {}

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
        if log_level > self.log_level:
            # the logged metric is more verbose than what we want to record.
            return
        for k, v in data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append((timestamp, log_level, v))
        self.most_recent_values.update(data)
        self.most_recent_timestamps.update({k: timestamp for k in data})
