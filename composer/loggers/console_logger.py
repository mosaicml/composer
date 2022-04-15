# Copyright 2021 MosaicML. All Rights Reserved.

"""Logs to the console."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, TextIO, Union

from composer.core.state import State
from composer.loggers.logger import LogLevel, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination

__all__ = ["ConsoleLogger"]


class ConsoleLogger(LoggerDestination):
    """Log to the console.

    .. note::

        This logger is automatically instainatied by the trainer via the ``console_log_level`` and
        ``console_log_stream`` options. This logger does not need to be created manually.
    
    Args:
        log_level (LogLevel | str | (State, LogLevel) -> bool, optional): The maximum log level which
            should be printed to the console. It can either be :class:`.LogLevel`, a string corresponding to a
            :class:`.LogLevel`, or a callable that takes the training :class:`.State` and the :class:`.LogLevel`
            and returns a boolean of whether this statement should be printed. (default: :attr:`.LogLevel.EPOCH`)
        stream (str | TextIO, optional): The stream to write to. If a string, it can either be ``'stdout'`` or ``'stderr'``
            (default: :attr:`sys.stdout`)
    """

    def __init__(
        self,
        log_level: Union[LogLevel, str, Callable[[State, LogLevel], bool]] = LogLevel.EPOCH,
        stream: Union[str, TextIO] = sys.stdout,
    ) -> None:
        if isinstance(log_level, str):
            log_level = LogLevel(log_level)
        if isinstance(log_level, LogLevel):
            def should_log(state: State, log_level: LogLevel, maximum_log_level: LogLevel = log_level):
                del state  # unused
                return log_level <= maximum_log_level
            self.should_log = should_log
        else:
            self.should_log = log_level
        if isinstance(stream, str):
            if stream.lower() == "stdout":
                stream = sys.stdout
            elif stream.lower() == "stderr":
                stream = sys.stderr
            else:
                raise ValueError("Invalid stream option: Should be 'stdout', 'stderr', or a TextIO-like object.")
        self.stream = stream

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        if not self.should_log(state, log_level):
            return
        data_str = format_log_data_value(data)
        print(
            f'[{log_level.name}][batch={int(state.timer.batch)}]: {data_str}',
            file=self.stream,
            flush=True,
        )
