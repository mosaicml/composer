from __future__ import annotations

import atexit
import os
import sys
from typing import Optional, TextIO

from composer.core.logging import Logger, LogLevel, RankZeroLoggerBackend, TLogData, format_log_data_value
from composer.core.state import State
from composer.loggers.logger_hparams import FileLoggerBackendHparams


class FileLoggerBackend(RankZeroLoggerBackend):

    def __init__(
        self,
        filename: str = 'stdout',
        *,
        buffer_size: int = 1,
        log_level: LogLevel = LogLevel.EPOCH,
        every_n_epochs: int = 1,
        every_n_batches: int = 1,
        flush_every_n_batches: int = 1,
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

    def batch_end(self, state: State, logger: Logger) -> None:
        assert self.file is not None
        if (state.step + 1) % self.hparams.flush_every_n_batches == 0 and self.file not in (sys.stdout, sys.stderr):
            self.file.flush()
            os.fsync(self.file.fileno())

    def _close_file(self) -> None:
        assert self.file is not None
        assert self.file not in (sys.stdout, sys.stderr)
        self.file.close()
