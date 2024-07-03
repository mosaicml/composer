# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs to a file."""

from __future__ import annotations

import os
import queue
import sys
import textwrap
from typing import TYPE_CHECKING, Any, Callable, Optional, TextIO

from composer.loggers.logger import Logger, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import FORMAT_NAME_WITH_DIST_TABLE, format_name_with_dist
from composer.utils.import_helpers import MissingConditionalImportError

if TYPE_CHECKING:
    from composer.core import State

__all__ = ['FileLogger']


class FileLogger(LoggerDestination):  # noqa: D101
    __doc__ = f"""Log data to a file.

    Example usage:
        .. testcode::

            from composer.loggers import FileLogger
            from composer.trainer import Trainer
            file_logger = FileLogger(
                filename="{{run_name}}/logs-rank{{rank}}.txt",
                buffer_size=1,
                flush_interval=50
            )
            trainer = Trainer(
                ...,
                loggers=[file_logger]
            )

        .. testcleanup::

            import os

            trainer.engine.close()

            path = os.path.join(trainer.state.run_name, "logs-rank0.txt")
            try:
                os.remove(file_logger.filename)
            except FileNotFoundError as e:
                pass

    Example output::

        [FIT][step=2]: {{ "logged_metric": "logged_value", }}
        [EPOCH][step=2]: {{ "logged_metric": "logged_value", }}
        [BATCH][step=2]: {{ "logged_metric": "logged_value", }}
        [EPOCH][step=3]: {{ "logged_metric": "logged_value", }}


    Args:
        filename (str, optional): Format string for the filename.

            The following format variables are available:

            {textwrap.indent(FORMAT_NAME_WITH_DIST_TABLE, prefix='            ')}

            .. note::

                When training with multiple devices (i.e. GPUs), ensure that ``'{{rank}}'`` appears in the format.
                Otherwise, multiple processes may attempt to write to the same file.

            Consider the following example when using default value of '{{run_name}}/logs-rank{{rank}}.txt':

            >>> file_logger = FileLogger(filename='{{run_name}}/logs-rank{{rank}}.txt')
            >>> trainer = Trainer(loggers=[file_logger], run_name='my-awesome-run')
            >>> file_logger.filename
            'my-awesome-run/logs-rank0.txt'

            Default: `'{{run_name}}/logs-rank{{rank}}.txt'`

        remote_file_name (str, optional): Format string for the logfile's name.

            The logfile will be periodically logged (according to the ``flush_interval``) as a file.
            The file name will be determined by this format string.

            .. seealso:: :doc:`Uploading Files</trainer/file_uploading>` for notes for file uploading.

            The same format variables for ``filename`` are available. Setting this parameter to ``None``
            (the default) will use the same format string as ``filename``. It is sometimes helpful to deviate
            from this default. For example, when ``filename`` contains an absolute path, it is recommended to
            set this parameter explicitely, so the absolute path does not appear in any remote file stores.

            Leading slashes (``'/'``) will be stripped.

            Default: ``None`` (which uses the same format string as ``filename``)
        capture_stdout (bool, optional): Whether to include the ``stdout``in ``filename``. (default: ``True``)
        capture_stderr (bool, optional): Whether to include the ``stderr``in ``filename``. (default: ``True``)
        buffer_size (int, optional): Buffer size. See :py:func:`open`.
            Default: ``1`` for line buffering.
        log_traces (bool, optional): Whether to log algorithm traces. See :class:`~.Engine` for more detail.
        flush_interval (int, optional): How frequently to flush the log to the file in batches
            Default: ``100``.
        overwrite (bool, optional): Whether to overwrite an existing logfile. (default: ``False``)
    """

    def __init__(
        self,
        filename: str = '{run_name}/logs-rank{rank}.txt',
        remote_file_name: Optional[str] = None,
        *,
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        buffer_size: int = 1,
        log_traces: bool = True,
        flush_interval: int = 100,
        overwrite: bool = False,
    ) -> None:
        self.filename_format = filename
        if remote_file_name is None:
            remote_file_name = filename.replace(os.path.sep, '/')
        self.remote_file_name_format = remote_file_name
        self.buffer_size = buffer_size
        self.should_log_traces = log_traces
        self.flush_interval = flush_interval
        self.is_batch_interval = False
        self.is_epoch_interval = False
        self.file: Optional[TextIO] = None
        self.overwrite = overwrite,
        self._queue: queue.Queue[str] = queue.Queue()
        self._run_name = None
        # Track whether the next line is on a newline
        # (and if so, then the prefix should be appended)
        self._is_newline = True
        self._closed = False

        if capture_stdout:
            sys.stdout.write = self._get_new_writer('[stdout]: ', sys.stdout.write)

        if capture_stderr:
            sys.stderr.write = self._get_new_writer('[stderr]: ', sys.stderr.write)

    def _get_new_writer(self, prefix: str, original_writer: Callable[[str], int]):
        """Returns a writer that intercepts calls to the ``original_writer``."""

        def new_write(s: str) -> int:
            if not self._closed:
                self.write(prefix, s)
            return original_writer(s)

        return new_write

    @property
    def filename(self) -> str:
        """The filename for the logfile."""
        if self._run_name is None:
            raise RuntimeError('The run name is not set. The engine should have been set on Event.INIT')
        name = format_name_with_dist(self.filename_format, run_name=self._run_name)

        return name

    @property
    def remote_file_name(self) -> str:
        """The remote file name for the logfile."""
        if self._run_name is None:
            raise RuntimeError('The run name is not set. The engine should have been set on Event.INIT')
        name = format_name_with_dist(self.remote_file_name_format, run_name=self._run_name)

        name.lstrip('/')

        return name

    def epoch_start(self, state: State, logger: Logger) -> None:
        # Flush any log calls that occurred during INIT or FIT_START
        self._flush_file(logger)

    def log_traces(self, traces: dict[str, Any]):
        if self.should_log_traces:
            for trace_name, trace in traces.items():
                trace_str = format_log_data_value(trace)
                self.write(
                    f'[trace]: {trace_name}:',
                    trace_str + '\n',
                )

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
        table = pd.DataFrame.from_records(columns=columns, data=rows).to_json(orient='split', index=False)
        self.write('[table]: ', f'{name}: {table}\n')

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        for metric_name, metric in metrics.items():
            metric_str = format_log_data_value(metric)
            self.write(
                f'[metric][batch={step}]: ',
                f'{metric_name}: {metric_str} \n',
            )

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        for hparam_name, hparam_value in hyperparameters.items():
            hparam_str = format_log_data_value(hparam_value)
            self.write(
                f'[hyperparameter]: ',
                f'{hparam_name}: {hparam_str} \n',
            )

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self._is_newline = True
        self._run_name = state.run_name
        if self.file is not None:
            raise RuntimeError('The file logger is already initialized')
        file_dirname = os.path.dirname(self.filename)
        if file_dirname:
            os.makedirs(file_dirname, exist_ok=True)
        mode = 'w+' if self.overwrite else 'x+'
        self.file = open(self.filename, mode, buffering=self.buffer_size)
        self._flush_queue()

    def batch_end(self, state: State, logger: Logger) -> None:
        assert self.file is not None
        if int(state.timestamp.batch) % self.flush_interval == 0:
            self._flush_file(logger)

    def eval_start(self, state: State, logger: Logger) -> None:
        # Flush any log calls that occurred during INIT when using the trainer in eval-only mode
        self._flush_file(logger)

    def epoch_end(self, state: State, logger: Logger) -> None:
        if int(state.timestamp.epoch) % self.flush_interval == 0:
            self._flush_file(logger)

    def write(self, prefix: str, s: str):
        """Write to the logfile.

        .. note::

            If the ``write`` occurs before the :attr:`.Event.INIT` event,
            the write will be enqueued, as the file is not yet open.

        Args:
            prefix (str): A prefix for each line in the logfile.
            s (str): The string to write. Each line will be prefixed with ``prefix``.
        """
        formatted_lines = []
        for line in s.splitlines(True):
            if self._is_newline:
                # Only print the prefix if it is a newline
                # and the line is not empty
                if line == os.linesep:
                    formatted_lines.append(line)
                else:
                    formatted_lines.append(f'{prefix}{line}')
                self._is_newline = False
            else:
                # Otherwise, append the line without the prefix
                formatted_lines.append(line)
            if line.endswith(os.linesep):
                # if the line ends with newline, record that the next
                # line should start with the prefix
                self._is_newline = True
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

    def _flush_file(self, logger: Logger) -> None:
        assert self.file is not None

        self._flush_queue()

        self.file.flush()
        os.fsync(self.file.fileno())
        logger.upload_file(self.remote_file_name, self.file.name, overwrite=True)

    def fit_end(self, state: State, logger: Logger) -> None:
        # Flush the file on fit_end, in case if was not flushed on epoch_end and the trainer is re-used
        # (which would defer when `self.close()` would be invoked)
        self._flush_file(logger)

    def close(self, state: State, logger: Logger) -> None:
        del state  # unused
        self._closed = True  # Stop intercepting calls to stdout/stderr
        if self.file is not None:
            self._flush_file(logger)
            self.file.close()
            self.file = None
