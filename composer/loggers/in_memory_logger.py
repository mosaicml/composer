# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import Dict, List, Tuple, Union

from composer.core.logging import LoggerCallback, LogLevel, TLogData
from composer.core.logging.logger import TLogDataValue
from composer.core.time import Timestamp


class InMemoryLogger(LoggerCallback):
    """Stores all logging calls in memory.

    Args:
        log_level (str or LogLevel, optional): Minimum LogLevel to record. Defaults to
            :attr:`LogLevel.BATCH`, which records everything.

    Attributes:
        data (dict): Mapping of a logged key to a
            (:class:`Timestamp`, :class:`LogLevel`, :class`~composer.core.logging.TLogDataValue`) tuple.
            This dictionary contains all logged data.
        most_recent_values (Dict[str, TLogData]): Mapping of a key to the most recent value for that key.
        most_recent_timestamps (Dict[str, Timestamp]): Mapping of a key to the :class:`Timestamp` of
            the last logging call for that key.
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
