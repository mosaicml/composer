# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to the console and without a progress bar."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Dict, TextIO, Union

import yaml

from composer.core.time import Time, TimeUnit
from composer.loggers.logger import Logger, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

if TYPE_CHECKING:
    from composer.core import State


class ConsoleLogger(LoggerDestination):
    """Log metrics to the console.

    .. note::

        This logger is automatically instantiated by the trainer via the ``log_to_console``,
        and ``console_stream`` options. This logger does not need to be created manually.

    Args:
        log_interval (int | str | Time): How frequently to log to console. (default: ``'1ep'``)
        stream (str | TextIO, optional): The console stream to use. If a string, it can either be ``'stdout'`` or
            ``'stderr'``. (default: :attr:`sys.stderr`)
        log_traces (bool): Whether to log traces or not. (default: ``False``)
    """

    def __init__(self,
                 log_interval: Union[int, str, Time] = '1ep',
                 stream: Union[str, TextIO] = sys.stderr,
                 log_traces: bool = False) -> None:

        if isinstance(log_interval, int):
            log_interval = Time(log_interval, TimeUnit.EPOCH)
        if isinstance(log_interval, str):
            log_interval = Time.from_timestring(log_interval)

        if log_interval.unit not in (TimeUnit.EPOCH, TimeUnit.BATCH):
            raise ValueError('The `console_log_interval` argument must have units of EPOCH or BATCH.')

        self.log_interval = log_interval
        # set the stream
        if isinstance(stream, str):
            if stream.lower() == 'stdout':
                stream = sys.stdout
            elif stream.lower() == 'stderr':
                stream = sys.stderr
            else:
                raise ValueError(f'stream must be one of ("stdout", "stderr", TextIO-like), got {stream}')

        self.should_log_traces = log_traces
        self.stream = stream
        self.hparams: Dict[str, Any] = {}
        self.hparams_already_logged_to_console: bool = False

    def log_traces(self, traces: Dict[str, Any]):
        if self.should_log_traces:
            for trace_name, trace in traces.items():
                trace_str = format_log_data_value(trace)
                self._log_to_console(f'[trace]: {trace_name}:' + trace_str + '\n')

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        # Lazy logging of hyperparameters.
        self.hparams.update(hyperparameters)

    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            self._log_to_console('*' * 30)
            self._log_to_console('Config:')
            self._log_to_console(yaml.dump(self.hparams))
            self._log_to_console('*' * 30)

    def epoch_end(self, state: State, logger: Logger) -> None:
        cur_epoch = int(state.timestamp.epoch)  # epoch gets incremented right before EPOCH_END
        unit = self.log_interval.unit

        if unit == TimeUnit.EPOCH and cur_epoch % int(self.log_interval) == 0:
            log_dict = {**state.train_metric_values}
            if state.total_loss_dict:
                log_dict.update(state.total_loss_dict)
            self.log_to_console(log_dict, prefix='Train ', state=state)

    def batch_end(self, state: State, logger: Logger) -> None:
        cur_batch = int(state.timestamp.batch)
        unit = self.log_interval.unit
        if unit == TimeUnit.BATCH and cur_batch % int(self.log_interval) == 0:
            log_dict = {**state.train_metric_values}
            if state.total_loss_dict:
                log_dict.update(state.total_loss_dict)
            self.log_to_console(log_dict, prefix='Train ', state=state)

    def eval_end(self, state: State, logger: Logger) -> None:
        self.log_to_console(state.eval_metric_values, prefix='Eval ', state=state)

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def predict_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def eval_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def _get_progress_string(self, state: State):
        if state.max_duration is None:
            training_progress = ''
        elif state.max_duration.unit == TimeUnit.EPOCH:
            cur_batch = int(state.timestamp.batch_in_epoch)
            cur_epoch = int(state.timestamp.epoch)
            if cur_batch == 0 and cur_epoch != 0:
                cur_epoch -= 1
                cur_batch = int(state.dataloader_len) if state.dataloader_len is not None else cur_batch
            if state.dataloader_len is None:
                curr_progress = f'[batch={cur_batch}]'
            else:
                total = int(state.dataloader_len)
                curr_progress = f'[batch={cur_batch}/{total}]'

            training_progress = f'[epoch={cur_epoch + 1}]{curr_progress}'
        else:
            unit = state.max_duration.unit
            curr_duration = int(state.timestamp.get(unit))
            total = state.max_duration.value
            training_progress = f'[{unit.name.lower()}={curr_duration}/{total}]'
        return training_progress

    def log_to_console(self, data: Dict[str, Any], state: State, prefix: str = '') -> None:
        # log to console
        training_progress = self._get_progress_string(state)
        log_str = f'{training_progress}:'
        for data_name, data in data.items():
            data_str = format_log_data_value(data)
            log_str += f'\n\t {prefix}{data_name}: {data_str}'
        self._log_to_console(log_str)

    def _log_to_console(self, log_str: str):
        """Logs to the console, avoiding interleaving with a progress bar."""
        # write directly to self.stream; no active progress bar
        print(log_str, file=self.stream, flush=True)
