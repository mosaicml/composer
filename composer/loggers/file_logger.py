# Copyright 2021 MosaicML. All Rights Reserved.

"""Logs to a file or to the terminal."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, TextIO

import yaml

from composer.core.logging import Logger, LoggerCallback, LogLevel, TLogData, format_log_data_value
from composer.core.state import State
from composer.core.time import Timestamp
from composer.utils import run_directory

__all__ = ["FileLogger"]


class FileLogger(LoggerCallback):
    """Logs to a file or to the terminal.

    Example usage:
        .. testcode::

            from composer.loggers import FileLogger
            from composer.trainer import Trainer
            from composer.core.logging import LogLevel
            logger = FileLogger(
                filename="log.txt",
                buffer_size=1,
                log_level=LogLevel.BATCH,
                log_interval=2,
                flush_interval=50
            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                optimizers=[optimizer],
                loggers=[logger]
            )

        .. testcleanup::

            import os
            from composer.utils.run_directory import get_run_directory

            path = os.path.join(get_run_directory(), "log.txt")
            try:
                os.remove(path)
            except FileNotFoundError as e:
                pass

    Example output::

        [FIT][step=2]: { "logged_metric": "logged_value", }
        [EPOCH][step=2]: { "logged_metric": "logged_value", }
        [BATCH][step=2]: { "logged_metric": "logged_value", }
        [EPOCH][step=3]: { "logged_metric": "logged_value", }


    Args:
        filename (str, optional): File to log to.
            Can be a filepath, ``"stdout"``, or ``"stderr"``. Default: ``"stdout"``.
            Filepaths should be specified relative to the
            :mod:`~.composer.utils.run_directory`.
        buffer_size (int, optional): Buffer size. See :py:func:`open`.
            Default: ``1`` for line buffering.
        log_level (LogLevel, optional):
            :class:`~.logger.LogLevel` (i.e. unit of resolution) at
            which to record. Default: :attr:`~.LogLevel.EPOCH`.
        log_interval (int, optional):
            Frequency to print logs. If ``log_level`` is :attr:`~.LogLevel.EPOCH`,
            logs will only be recorded every n epochs. If ``log_level`` is
            :attr:`~.LogLevel.BATCH`, logs will be printed every n batches.  Otherwise, if
            ``log_level`` is :attr:`~.LogLevel.FIT`, this parameter is ignored, as calls
            at the :attr:`~.LogLevel.FIT` log level are always recorded. Default: ``1``.
        flush_interval (int, optional): How frequently to flush the log to the file,
            relative to the ``log_level``. For example, if the ``log_level`` is
            :attr:`~.LogLevel.EPOCH`, then the logfile will be flushed every n epochs.  If
            the ``log_level`` is :attr:`~.LogLevel.BATCH`, then the logfile will be
            flushed every n batches. Default: ``100``.
    """

    def __init__(
        self,
        filename: str = 'stdout',
        *,
        buffer_size: int = 1,
        log_level: LogLevel = LogLevel.EPOCH,
        log_interval: int = 1,
        flush_interval: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.buffer_size = buffer_size
        self.log_level = log_level
        self.log_interval = log_interval
        self.flush_interval = flush_interval
        self.is_batch_interval = False
        self.is_epoch_interval = False
        self.file: Optional[TextIO] = None
        self.config = config

    def batch_start(self, state: State, logger: Logger) -> None:
        self.is_batch_interval = (int(state.timer.batch) + 1) % self.log_interval == 0

    def epoch_start(self, state: State, logger: Logger) -> None:
        self.is_epoch_interval = (int(state.timer.epoch) + 1) % self.log_interval == 0
        # Flush any log calls that occurred during INIT or FIT_START
        self._flush_file()

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        if log_level == LogLevel.FIT:
            return True  # fit is always logged
        if log_level == LogLevel.EPOCH:
            if self.log_level < LogLevel.EPOCH:
                return False
            if self.log_level > LogLevel.EPOCH:
                return True
            return self.is_epoch_interval
        if log_level == LogLevel.BATCH:
            if self.log_level < LogLevel.BATCH:
                return False
            if self.log_level > LogLevel.BATCH:
                return True
            return self.is_batch_interval
        raise ValueError(f"Unknown log level: {log_level}")

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
        data_str = format_log_data_value(data)
        if self.file is None:
            raise RuntimeError("Attempted to log before self.init() or after self.close()")
        print(f"[{log_level.name}][step={int(timestamp.batch)}]: {data_str}", file=self.file, flush=False)

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self.file is not None:
            raise RuntimeError("The file logger is already initialized")
        if self.filename == "stdout":
            self.file = sys.stdout
        elif self.filename == "stderr":
            self.file = sys.stderr
        else:
            self.file = open(os.path.join(run_directory.get_run_directory(), self.filename),
                             "x+",
                             buffering=self.buffer_size)
        if self.config is not None:
            print("Config", file=self.file)
            print("-" * 30, file=self.file)
            yaml.safe_dump(self.config, stream=self.file)
            print("-" * 30, file=self.file)
            print(file=self.file)

    def batch_end(self, state: State, logger: Logger) -> None:
        del logger  # unused
        assert self.file is not None
        if self.log_level == LogLevel.BATCH and int(state.timer.batch) % self.flush_interval == 0:
            self._flush_file()

    def eval_start(self, state: State, logger: Logger) -> None:
        # Flush any log calls that occurred during INIT when using the trainer in eval-only mode
        self._flush_file()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if self.log_level > LogLevel.EPOCH or self.log_level == LogLevel.EPOCH and int(
                state.timer.epoch) % self.flush_interval == 0:
            self._flush_file()

    def _flush_file(self) -> None:
        assert self.file is not None
        if self.file not in (sys.stdout, sys.stderr):
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self) -> None:
        if self.file is not None:
            if self.file not in (sys.stdout, sys.stderr):
                self._flush_file()
                self.file.close()
            self.file = None
