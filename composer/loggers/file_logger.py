# Copyright 2021 MosaicML. All Rights Reserved.

"""Logs to a file or to the terminal."""

from __future__ import annotations

import os
import queue
import sys
from typing import Any, Callable, Dict, Optional, TextIO

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import run_directory

__all__ = ["FileLogger"]


class FileLogger(LoggerDestination):
    """Log data to a file.

    Example usage:
        .. testcode::

            from composer.loggers import FileLogger, LogLevel
            from composer.trainer import Trainer
            file_logger = FileLogger(
                filename="log.txt",
                buffer_size=1,
                log_level=LogLevel.BATCH,
                log_interval=2,
                flush_interval=50
            )
            trainer = Trainer(
                ...,
                loggers=[file_logger]
            )

        .. testcleanup::

            import os

            file_logger.close()

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
        filename (str): Filepath to log to, relative to the :mod:`~.composer.utils.run_directory`.
        capture_stdout (bool, optional): Whether to include the ``stdout``in ``filename``. (default: ``True``)
        capture_stderr (bool, optional): Whether to include the ``stderr``in ``filename``. (default: ``True``)
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
        filename: str,
        *,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        buffer_size: int = 1,
        log_level: LogLevel = LogLevel.EPOCH,
        log_interval: int = 1,
        flush_interval: int = 100,
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
        self._queue: queue.Queue[str] = queue.Queue()
        self._original_stdout_write = sys.stdout.write
        self._original_stderr_write = sys.stderr.write

        if capture_stdout:
            sys.stdout.write = self._get_new_writer("[stdout]: ", self._original_stdout_write)

        if capture_stderr:
            sys.stderr.write = self._get_new_writer("[stderr]: ", self._original_stderr_write)

    def _get_new_writer(self, prefix: str, original_writer: Callable[[str], int]):
        """Returns a writer that intercepts calls to the ``original_writer``."""

        def new_write(s: str) -> int:
            self.write(prefix, s)
            return original_writer(s)

        return new_write

    def batch_start(self, state: State, logger: Logger) -> None:
        self.is_batch_interval = (int(state.timer.batch) + 1) % self.log_interval == 0

    def epoch_start(self, state: State, logger: Logger) -> None:
        self.is_epoch_interval = (int(state.timer.epoch) + 1) % self.log_interval == 0
        # Flush any log calls that occurred during INIT or FIT_START
        self._flush_file()

    def _will_log(self, log_level: LogLevel) -> bool:
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

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        if not self._will_log(log_level):
            return
        data_str = format_log_data_value(data)
        self.write(
            f'[{log_level.name}][batch={int(state.timer.batch)}]: ',
            data_str + "\n",
        )

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self.file is not None:
            raise RuntimeError("The file logger is already initialized")
        self.file = open(
            os.path.join(run_directory.get_run_directory(), self.filename),
            "x+",
            buffering=self.buffer_size,
        )
        self._flush_queue()

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

    def write(self, prefix: str, s: str):
        """Write to the logfile.

        .. note::

            If the ``write`` occurs before the :attr:`~composer.core.event.Event.INIT` event,
            the write will be enqueued, as the file is not yet open.

        Args:
            prefix (str): A prefix for each line in the logfile.
            s (str): The string to write. Each line will be prefixed with ``prefix``.
        """
        formatted_lines = []
        for line in s.splitlines(True):
            if line == os.linesep:
                # If it's an empty line, don't print the prefix
                formatted_lines.append(line)
            else:
                formatted_lines.append(f"{prefix}{line}")
        formatted_s = ''.join(formatted_lines)
        if self.file is None:
            self._queue.put_nowait(formatted_s)
        else:
            # Flush the queue, so all prints will be in order
            self._flush_queue()
            # Then, write to the file
            print(formatted_s, file=self.file, flush=False, end='')

    def _flush_queue(self):
        while True:
            try:
                s = self._queue.get_nowait()
            except queue.Empty:
                break
            print(s, file=self.file, flush=False, end='')

    def _flush_file(self) -> None:
        assert self.file is not None

        self._flush_queue()

        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        if self.file is not None:
            sys.stdout.write = self._original_stdout_write
            sys.stderr.write = self._original_stderr_write
            self._flush_file()
            self.file.close()
            self.file = None
