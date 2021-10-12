# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import atexit
import os
import sys
from typing import Any, Dict, Optional, TextIO

import yaml

from composer.core.logging import Logger, LogLevel, RankZeroLoggerBackend, TLogData, format_log_data_value
from composer.core.state import State
from composer.loggers.logger_hparams import FileLoggerBackendHparams


class FileLoggerBackend(RankZeroLoggerBackend):
    """Log to a file.

    It prints logs to a file
    (or terminal, defaults to stdout) as specified by :param filename:.
    The :param log_level: 
    (defaults to :attr:`~composore.core.logging.logger.LogLevel.EPOCH`)
    controls the verbosity of the logging.

    Example output:
        ```
        [FIT][step=2]: { "logged_metric": "logged_value", }
        [EPOCH][step=2]: { "logged_metric": "logged_value", }
        [BATCH][step=2]: { "logged_metric": "logged_value", }
        [EPOCH][step=3]: { "logged_metric": "logged_value", }
        ```


    Args:
        filename (str, optional): File to log to. Can be a filepath, 'stdout', or 'stderr'. Defaults to 'stdout'.
        buffer_size (int, optional): Buffer size. See
            https://docs.python.org/3/library/functions.html#open. Defaults to 1
            for line buffering.
        log_level (LogLevel, optional): Maximum
            :class:`~composore.core.logging.logger.LogLevel`. to record.
            Defaults to :attr:`~composore.core.logging.logger.LogLevel.EPOCH`.
        every_n_epochs (int, optional):
            Frequency to print :attr:`~composore.core.logging.logger.LogLevel.EPOCH` logs.
            Defaults to 1 (every epoch).
        every_n_batches (int, optional):
            Frequency to print :attr:`~composore.core.logging.logger.LogLevel.BATCH` logs.
            Defaults to 1 (every batch).
        flush_every_n_batches (int, optional): How frequently to flush the log to the file.
            Defaults to 1 (every batch).
    """

    def __init__(
        self,
        filename: str = 'stdout',
        *,
        buffer_size: int = 1,
        log_level: LogLevel = LogLevel.EPOCH,
        every_n_epochs: int = 1,
        every_n_batches: int = 1,
        flush_every_n_batches: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hparams = FileLoggerBackendHparams(
            filename=filename,
            buffer_size=buffer_size,
            log_level=log_level,
            every_n_epochs=every_n_epochs,
            every_n_batches=every_n_batches,
            flush_every_n_batches=flush_every_n_batches,
        )
        self.file: Optional[TextIO] = None
        self.config = config

    def _will_log(self, state: State, log_level: LogLevel) -> bool:
        if log_level > self.hparams.log_level:
            return False
        if log_level >= LogLevel.EPOCH and state.epoch % self.hparams.every_n_epochs != 0:
            return False
        if log_level >= LogLevel.BATCH and (state.step + 1) % self.hparams.every_n_batches != 0:
            return False
        return True

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        data_str = format_log_data_value(data)
        print(f"[{log_level.name}][step={step}]: {data_str}", file=self.file)

    def _training_start(self, state: State, logger: Logger) -> None:
        if self.hparams.filename == "stdout":
            self.file = sys.stdout
        elif self.hparams.filename == "stderr":
            self.file = sys.stderr
        else:
            self.file = open(self.hparams.filename, "x+", buffering=self.hparams.buffer_size)
            atexit.register(self._close_file)
        if self.config is not None:
            print("Config", file=self.file)
            print("-" * 30, file=self.file)
            yaml.safe_dump(self.config, stream=self.file)
            print("-" * 30, file=self.file)
            print(file=self.file)

    def batch_end(self, state: State, logger: Logger) -> None:
        assert self.file is not None
        if (state.step + 1) % self.hparams.flush_every_n_batches == 0 and self.file not in (sys.stdout, sys.stderr):
            self.file.flush()
            os.fsync(self.file.fileno())

    def _close_file(self) -> None:
        assert self.file is not None
        assert self.file not in (sys.stdout, sys.stderr)
        self.file.close()
