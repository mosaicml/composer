# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to the console and without a progress bar."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Optional, Sequence, TextIO, Union

import numpy as np
import yaml

from composer.core.time import Time, TimeUnit
from composer.loggers.logger import Logger, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

if TYPE_CHECKING:
    from composer.core import State

# We use deciles here, so 11 events because deciles including 0.
NUM_EVAL_LOGGING_EVENTS = 11


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

    def __init__(
        self,
        log_interval: Union[int, str, Time] = '1ba',
        stream: Union[str, TextIO] = sys.stderr,
        log_traces: bool = False,
    ) -> None:

        log_interval = Time.from_input(log_interval, TimeUnit.EPOCH)
        self.last_logged_batch = 0

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
        self.hparams: dict[str, Any] = {}
        self.hparams_already_logged_to_console: bool = False
        self.logged_metrics: dict[str, float] = {}
        self.eval_batch_idxs_to_log: Sequence[int] = []
        self.tables: dict[str, str] = {}

    def log_traces(self, traces: dict[str, Any]):
        if self.should_log_traces:
            for trace_name, trace in traces.items():
                trace_str = format_log_data_value(trace)
                self._log_to_console(f'[trace]: {trace_name}:' + trace_str + '\n')

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        # Lazy logging of hyperparameters.
        self.hparams.update(hyperparameters)

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
        table = pd.DataFrame.from_records(data=rows, columns=columns).to_json(orient='split', index=False)
        self.tables[name] = str(table)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        del step
        # Lazy logging of metrics.
        # Stores all metrics logged until they are cleared with a log_to_console call
        self.logged_metrics.update(metrics)

    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            self._log_to_console('*' * 30)
            self._log_to_console('Config:')
            self._log_to_console(yaml.dump(self.hparams))
            self._log_to_console('*' * 30)

    def epoch_end(self, state: State, logger: Logger) -> None:
        cur_epoch = int(state.timestamp.epoch)  # epoch gets incremented right before EPOCH_END
        unit = self.log_interval.unit

        if unit == TimeUnit.EPOCH and (cur_epoch % int(self.log_interval) == 0 or self.last_logged_batch == 0):
            self.log_to_console(self.logged_metrics, prefix='Train ', state=state)
            self.last_logged_batch = int(state.timestamp.batch)
            self.logged_metrics = {}  # Clear logged metrics.

    def batch_end(self, state: State, logger: Logger) -> None:
        cur_batch = int(state.timestamp.batch)
        unit = self.log_interval.unit
        if unit == TimeUnit.BATCH and (cur_batch % int(self.log_interval) == 0 or self.last_logged_batch == 0):
            self.log_to_console(self.logged_metrics, prefix='Train ', state=state)
            self.last_logged_batch = cur_batch
            self.logged_metrics = {}  # Clear logged metrics.

    def iteration_end(self, state: State, logger: Logger) -> None:
        cur_iteration = int(state.timestamp.iteration)  # iteration gets incremented right before ITERATION_END
        unit = self.log_interval.unit

        if unit == TimeUnit.ITERATION and (cur_iteration % int(self.log_interval) == 0 or self.last_logged_batch == 0):
            self.log_to_console(self.logged_metrics, prefix='Train ', state=state)
            self.last_logged_batch = int(state.timestamp.batch)
            self.logged_metrics = {}  # Clear logged metrics.

    def fit_end(self, state: State, logger: Logger) -> None:
        # Always clear logged metrics so they don't get logged in a subsequent eval call.
        cur_batch = int(state.timestamp.batch)
        if self.last_logged_batch != cur_batch:
            self.log_to_console(
                self.logged_metrics,
                prefix='Train ',
                state=state,
            )  # log at the end of training if you didn't just log

        self.logged_metrics = {}

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        cur_batch = int(state.eval_timestamp.batch)
        if cur_batch in self.eval_batch_idxs_to_log:
            self.log_to_console({}, prefix='Eval ', state=state, is_train=False)

    def eval_end(self, state: State, logger: Logger) -> None:
        # Log to the console at the end of eval no matter what log interval is selected.
        self.log_to_console(self.logged_metrics, prefix='Eval ', state=state, is_train=False)
        self.logged_metrics = {}

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def predict_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def eval_start(self, state: State, logger: Logger) -> None:
        self.logged_metrics = {}  # Clear logged metrics before eval so they don't get logged subsequently.
        total_eval_batches = self._get_total_eval_batches(state)
        deciles = np.linspace(0, 1, NUM_EVAL_LOGGING_EVENTS)
        batch_idxs = np.arange(1, total_eval_batches + 1)
        if total_eval_batches < NUM_EVAL_LOGGING_EVENTS:
            self.eval_batch_idxs_to_log = list(batch_idxs)
        else:
            self.eval_batch_idxs_to_log = list(np.quantile(batch_idxs, deciles).round().astype(dtype=int))
        # Remove index of last batch, so that we don't print progress at end of last batch and then
        # at eval end.
        last_batch_idx = total_eval_batches
        if last_batch_idx in self.eval_batch_idxs_to_log:  # eval_batch_idxs_to_log may be empty
            self.eval_batch_idxs_to_log.remove(last_batch_idx)
        if not self.hparams_already_logged_to_console:
            self.hparams_already_logged_to_console = True
            self._log_hparams_to_console()

    def _get_eval_progress_string(self, state: State):
        eval_batch = state.eval_timestamp.batch.value
        eval_dataloader_label = state.dataloader_label
        total_eval_batches = self._get_total_eval_batches(state)
        curr_progress = f'[Eval batch={eval_batch}/{total_eval_batches}] Eval on {eval_dataloader_label} data'
        return curr_progress

    def _get_total_eval_batches(self, state: State) -> int:
        cur_evaluator = [evaluator for evaluator in state.evaluators if evaluator.label == state.dataloader_label][0]
        total_eval_batches = int(
            state.dataloader_len,
        ) if state.dataloader_len is not None else cur_evaluator.subset_num_batches
        # To please pyright. Based on _set_evaluator_interval_and_subset_num_batches, total_eval_batches can't be None
        assert total_eval_batches is not None
        return total_eval_batches

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

    def log_to_console(self, data: dict[str, Any], state: State, prefix: str = '', is_train=True) -> None:
        # log to console
        if is_train:
            progress = self._get_progress_string(state)
        else:
            progress = self._get_eval_progress_string(state)
        log_str = f'{progress}' + (':' if len(data) > 0 else '')
        for data_name, data in data.items():
            data_str = format_log_data_value(data)
            log_str += f'\n\t {prefix}{data_name}: {data_str}'
        for table_name, table in self.tables.items():
            log_str += f'\n\t {prefix}{table_name}: {table}'
        self._log_to_console(log_str)

    def _log_to_console(self, log_str: str):
        """Logs to the console, avoiding interleaving with a progress bar."""
        # write directly to self.stream; no active progress bar
        print(log_str, file=self.stream, flush=True)
