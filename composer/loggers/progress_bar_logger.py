# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to the console and show a progress bar."""

from __future__ import annotations

import os
import string
import sys
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TextIO, Union

import tqdm.auto

from composer.core.state import State
from composer.core.time import Timestamp, TimeUnit
from composer.loggers.logger import Logger, LogLevel, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

__all__ = ['ProgressBarLogger']


class MetricFormatter(string.Formatter):

    def get_field(self, field_name: str, args: Sequence[Any], kwargs: Mapping[str, Any]) -> Any:
        try:
            obj, arg_used = super().get_field(field_name, args, kwargs)
            return format_log_data_value(obj), arg_used
        except (AttributeError, KeyError):
            return '{' + str(field_name) + '}', None

    def format_field(self, value: Any, format_spec: str) -> Any:
        return super().format_field(value, format_spec)


class _ProgressBar:

    def __init__(
        self,
        total: Optional[int],
        position: int,
        bar_format: str,
        file: TextIO,
        timestamp_key: str,
        dataloader_label_to_metrics: Optional[Dict[str, Union[str, bool]]] = None,
        is_train: bool = False,
        unit: str = 'it',
    ) -> None:
        self.position = position
        self.timestamp_key = timestamp_key
        self.file = file
        self.dataloader_label_to_metrics = dataloader_label_to_metrics
        is_atty = os.isatty(self.file.fileno())
        self.is_train = is_train
        self.pbar = tqdm.auto.tqdm(
            total=total,
            position=position,
            bar_format=bar_format,
            file=file,
            ncols=None if is_atty else 120,
            dynamic_ncols=is_atty,
            # We set `leave=False` so TQDM does not jump around, but we emulate `leave=True` behavior when closing
            # by printing a dummy newline and refreshing to force tqdm to print to a stale line
            # But on k8s, we need `leave=True`, as it would otherwise overwrite the pbar in place
            leave=not is_atty,
            postfix={},
            unit=unit,
        )

    def update_metrics(self, state: State):
        dataloader_label = state.dataloader_label
        assert dataloader_label is not None
        if self.dataloader_label_to_metrics is not None and dataloader_label in self.dataloader_label_to_metrics:
            entry = self.dataloader_label_to_metrics[dataloader_label]
            if isinstance(entry, str):
                metric_string = entry
                metrics = state.current_metrics.get(dataloader_label, {})
                ft = MetricFormatter()
                metric_string = ft.format(metric_string, state=state)
                metric_string = ft.format(metric_string, **metrics)
                self.pbar.set_postfix_str(metric_string)
                return
            elif not entry:
                return
        metrics_dict = state.current_metrics.get(dataloader_label, {})
        formatted_data = {k: format_log_data_value(v) for (k, v) in metrics_dict.items()}
        if self.is_train:
            formatted_data['loss/train'] = format_log_data_value(state.loss)
        self.pbar.set_postfix(formatted_data)

    def update(self, n=1):
        self.pbar.update(n=n)

    def update_to_timestamp(self, timestamp: Timestamp):
        n = int(getattr(timestamp, self.timestamp_key))
        n = n - self.pbar.n
        self.pbar.update(int(n))

    def close(self):
        if self.position != 0:
            # Force a (potentially hidden) progress bar to re-render itself
            # Don't render the dummy pbar (at position 0), since that will clear a real pbar (at position 1)
            self.pbar.refresh()
        # Create a newline that will not be erased by leave=False. This allows for the finished pbar to be cached in the terminal
        # This emulates `leave=True` without progress bar jumping
        print('', file=self.file, flush=True)

        self.pbar.close()

    def state_dict(self) -> Dict[str, Any]:
        pbar_state = self.pbar.format_dict

        return {
            'total': pbar_state['total'],
            'position': self.position,
            'bar_format': pbar_state['bar_format'],
            'n': pbar_state['n'],
            'timestamp_key': self.timestamp_key,
        }


class ProgressBarLogger(LoggerDestination):
    """Log metrics to the console and optionally show a progress bar.

    .. note::

        This logger is automatically instainatied by the trainer via the ``progress_bar``, ``log_to_console``,
        ``log_level``, and ``console_stream`` options. This logger does not need to be created manually.

    `TQDM <https://github.com/tqdm/tqdm>`_ is used to display progress bars.

    During training, the progress bar logs the batch and training loss.
    During validation, the progress bar logs the batch and validation accuracy.

    Example progress bar output::

        Epoch 1: 100%|██████████| 64/64 [00:01<00:00, 53.17it/s, loss/train=2.3023]
        Epoch 1 (val): 100%|██████████| 20/20 [00:00<00:00, 100.96it/s, accuracy/val=0.0995]

    Args:
        progress_bar (bool, optional): Whether to show a progress bar. (default: ``True``)
        bar_format (str, optional): The format string passed into the tqdm progress bar. More info
            `here <https://tqdm.github.io/docs/tqdm/#__init__>`_. (default: ``'{l_bar}{bar:25}{r_bar}{bar:-1b}'``)
        dataloader_label_to_metrics (Dict[str, Union[bool, str]], optional): A dictionary specifying which metrics to
            log in the progress bar.

            If this parameter is ``None``, then the progress bar will print all metrics on all progress bars. If
            ``dataloader_label_to_metrics[dl_label]`` is ``True``, it prints all metrics for that dataloader. Otherwise,
            print all the metrics listed in the string ``dataloader_label_to_metrics[dl_label]``

            If ``dataloader_label_to_metrics[dl_label]`` is a string, the progress bar can access state variables and
            metrics for the relevant dataloader. Example: `'loss={state.loss}, accuracy={Accuracy}'`
        log_to_console (bool, optional): Whether to print logging statements to the console. (default: ``None``)

            The default behavior (when set to ``None``) only prints logging statements when ``progress_bar`` is
            ``False``.
        console_log_level (LogLevel | str | (State, LogLevel) -> bool, optional): The maximum log level for which statements
            should be printed. (default: :attr:`.LogLevel.EPOCH`)

            It can either be :class:`.LogLevel`, a string corresponding to a :class:`.LogLevel`, or a callable that
            takes the training :class:`.State` and the :class:`.LogLevel` and returns a boolean of whether this
            statement should be printed.

            This parameter has no effect if ``log_to_console`` is ``False`` or is unspecified when ``progress_bar`` is
            ``True``.
        stream (str | TextIO, optional): The console stream to use. If a string, it can either be ``'stdout'`` or
            ``'stderr'``. (default: :attr:`sys.stderr`)
    """

    def __init__(
        self,
        progress_bar: bool = True,
        bar_format: str = '{l_bar}{bar:25}{r_bar}{bar:-1b}',
        dataloader_label_to_metrics: Optional[Dict[str, Union[bool, str]]] = None,
        log_to_console: Optional[bool] = None,
        console_log_level: Union[LogLevel, str, Callable[[State, LogLevel], bool]] = LogLevel.EPOCH,
        stream: Union[str, TextIO] = sys.stderr,
    ) -> None:

        self._show_pbar = progress_bar
        # The dummy pbar is to fix issues when streaming progress bars over k8s, where the progress bar in position 0
        # doesn't update until it is finished.
        # Need to have a dummy progress bar in position 0, so the "real" progress bars in position 1 doesn't jump around
        self.dummy_pbar: Optional[_ProgressBar] = None
        self.dataloader_label_to_metrics = dataloader_label_to_metrics
        self.bar_format = bar_format
        self.train_pbar: Optional[_ProgressBar] = None
        self.eval_pbar: Optional[_ProgressBar] = None

        if isinstance(console_log_level, str):
            console_log_level = LogLevel(console_log_level)

        if log_to_console is None:
            log_to_console = not progress_bar

        if not log_to_console:
            # never log to console
            self.should_log = lambda state, ll: False
        else:
            # set should_log to a Callable[[State, LogLevel], bool]
            if isinstance(console_log_level, LogLevel):
                self.should_log = lambda state, ll: ll <= console_log_level
            else:
                self.should_log = console_log_level

        # set the stream
        if isinstance(stream, str):
            if stream.lower() == 'stdout':
                stream = sys.stdout
            elif stream.lower() == 'stderr':
                stream = sys.stderr
            else:
                raise ValueError(f'stream must be one of ("stdout", "stderr", TextIO-like), got {stream}')

        self.stream = stream

    @property
    def show_pbar(self) -> bool:
        return self._show_pbar and dist.get_local_rank() == 0

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]) -> None:
        # log to progress bar
        current_pbar = self.eval_pbar if self.eval_pbar is not None else self.train_pbar
        if current_pbar:
            # Logging outside an epoch
            current_pbar.update_metrics(state)

        # log to console
        if self.should_log(state, log_level):
            data_str = format_log_data_value(data)
            if state.max_duration is None:
                training_progress = ''
            elif state.max_duration.unit == TimeUnit.EPOCH:
                if state.dataloader_len is None:
                    curr_progress = f'[batch={int(state.timestamp.batch_in_epoch)}]'
                else:
                    total = int(state.dataloader_len)
                    curr_progress = f'[batch={int(state.timestamp.batch_in_epoch)}/{total}]'

                training_progress = f'[epoch={int(state.timestamp.epoch)}]{curr_progress}'
            else:
                unit = state.max_duration.unit
                curr_duration = int(state.timestamp.get(unit))
                total = state.max_duration.value
                training_progress = f'[{unit.name.lower()}={curr_duration}/{total}]'

            log_str = f'[{log_level.name}]{training_progress}: {data_str}'
            self.log_to_console(log_str)

    def log_to_console(self, log_str: str):
        """Logs to the console, avoiding interleaving with a progress bar."""
        current_pbar = self.eval_pbar if self.eval_pbar is not None else self.train_pbar
        if current_pbar:
            # use tqdm.write to avoid interleaving
            current_pbar.pbar.write(log_str)
        else:
            # write directly to self.stream; no active progress bar
            print(log_str, file=self.stream, flush=True)

    def _build_pbar(self, state: State, is_train: bool) -> _ProgressBar:
        """Builds a pbar.

        *   If ``state.max_duration.unit`` is :attr:`.TimeUnit.EPOCH`, then a new progress bar will be created for each epoch.
            Mid-epoch evaluation progress bars will be labeled with the batch and epoch number.
        *   Otherwise, one progress bar will be used for all of training. Evaluation progress bars will be labeled
            with the time (in units of ``max_duration.unit``) at which evaluation runs.
        """
        # Always using position=1 to avoid jumping progress bars
        position = 1
        desc = f'{state.dataloader_label:15}'
        max_duration_unit = None if state.max_duration is None else state.max_duration.unit

        if max_duration_unit == TimeUnit.EPOCH or max_duration_unit is None:
            total = int(state.dataloader_len) if state.dataloader_len is not None else None
            timestamp_key = 'batch_in_epoch'

            unit = TimeUnit.BATCH
            n = state.timestamp.epoch.value
            if state.timestamp.batch_in_epoch == 0 and not is_train:
                # epochwise eval results refer to model from previous epoch (n-1)
                n -= 1
            if state.timestamp.batch_in_epoch != 0 and not is_train:
                # For evaluation mid-epoch, show the total batch count
                desc += f'Batch {int(state.timestamp.batch):3}'
            else:
                desc += f'Epoch {n:3}'
            desc += ': '

        else:
            if is_train:
                assert state.max_duration is not None, 'max_duration should be set if training'
                unit = max_duration_unit
                total = state.max_duration.value
                # pad for the expected length of an eval pbar -- which is 14 characters (see the else logic below)
                desc = desc.ljust(len(desc) + 14)
            else:
                unit = TimeUnit.BATCH
                total = int(state.dataloader_len) if state.dataloader_len is not None else None
                value = int(state.timestamp.get(max_duration_unit))
                # Longest unit name is sample (6 characters)
                desc += f'{max_duration_unit.name.capitalize():6} {value:5}: '

            timestamp_key = unit.name.lower()

        return _ProgressBar(
            file=self.stream,
            total=total,
            position=position,
            dataloader_label_to_metrics=self.dataloader_label_to_metrics,
            is_train=is_train,
            bar_format=desc + ' ' + self.bar_format,
            unit=unit.value.lower(),
            timestamp_key=timestamp_key,
        )

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self.dummy_pbar = _ProgressBar(
            file=self.stream,
            position=0,
            total=1,
            bar_format='{bar:-1b}',
            timestamp_key='',
        )

    def epoch_start(self, state: State, logger: Logger) -> None:
        if self.show_pbar and not self.train_pbar:
            self.train_pbar = self._build_pbar(state=state, is_train=True)

    def eval_start(self, state: State, logger: Logger) -> None:
        if self.show_pbar:
            self.eval_pbar = self._build_pbar(state, is_train=False)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.train_pbar:
            self.train_pbar.update_metrics(state)
            self.train_pbar.update_to_timestamp(state.timestamp)

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if self.eval_pbar:
            self.eval_pbar.update_metrics(state)
            self.eval_pbar.update_to_timestamp(state.eval_timestamp)

    def epoch_end(self, state: State, logger: Logger) -> None:
        # Only close progress bars at epoch end if the duration is in epochs, since
        # a new pbar will be created for each epoch
        # If the duration is in other units, then one progress bar will be used for all of training.
        assert state.max_duration is not None, 'max_duration should be set'
        if self.train_pbar and state.max_duration.unit == TimeUnit.EPOCH:
            self.train_pbar.update_metrics(state)
            self.train_pbar.close()
            self.train_pbar = None

    def close(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        # Close any open progress bars
        if self.eval_pbar:
            self.eval_pbar.close()
            self.eval_pbar = None
        if self.train_pbar:
            self.train_pbar.close()
            self.train_pbar = None
        if self.dummy_pbar:
            self.dummy_pbar.close()
            self.dummy_pbar = None

    def eval_end(self, state: State, logger: Logger) -> None:
        if self.eval_pbar:
            self.eval_pbar.update_metrics(state)
            self.eval_pbar.close()
            self.eval_pbar = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_pbar': self.train_pbar.state_dict() if self.train_pbar else None,
            'eval_pbar': self.eval_pbar.state_dict() if self.eval_pbar else None,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state['train_pbar']:
            n = state['train_pbar'].pop('n')
            train_pbar = self._ensure_backwards_compatibility(state['train_pbar'])
            self.train_pbar = _ProgressBar(file=self.stream, **train_pbar)
            self.train_pbar.update(n=n)
        if state['eval_pbar']:
            n = state['train_pbar'].pop('n')
            eval_pbar = self._ensure_backwards_compatibility(state['eval_pbar'])
            self.eval_pbar = _ProgressBar(file=self.stream, **eval_pbar)
            self.eval_pbar.update(n=n)

    def _ensure_backwards_compatibility(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # ensure backwards compatible with mosaicml<=v0.8.0 checkpoints

        state.pop('epoch_style', None)

        # old checkpoints do not have timestamp_key
        if 'timestamp_key' not in state:
            if 'unit' not in state:
                raise ValueError('Either unit or timestamp_key must be in pbar state of checkpoint.')
            unit = state['unit']
            assert isinstance(unit, TimeUnit)

            state['timestamp_key'] = unit.name.lower()

        # new format expects unit as str, not TimeUnit
        if 'unit' in state and isinstance(state['unit'], TimeUnit):
            state['unit'] = state['unit'].value.lower()

        return state