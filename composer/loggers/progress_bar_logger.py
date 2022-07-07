# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to the console and show a progress bar."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

import tqdm.auto

from composer.core.state import State
from composer.core.time import Timestamp, TimeUnit
from composer.loggers.logger import Logger, LogLevel, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

__all__ = ['ProgressBarLogger']

_IS_TRAIN_TO_KEYS_TO_LOG = {True: ['loss/train'], False: ['metrics/eval/Accuracy']}


class _ProgressBar:

    def __init__(
        self,
        total: Optional[int],
        position: int,
        bar_format: str,
        file: TextIO,
        metrics: Dict[str, Any],
        keys_to_log: List[str],
        timestamp_key: str,
        unit: TimeUnit = TimeUnit.EPOCH,
    ) -> None:
        self.keys_to_log = keys_to_log
        self.metrics = metrics
        self.position = position
        self.unit = unit
        self.timestamp_key = timestamp_key
        self.file = file
        self.pbar = tqdm.auto.tqdm(
            total=total,
            position=position,
            bar_format=bar_format,
            file=file,
            dynamic_ncols=True,
            # We set `leave=False` so TQDM does not jump around, but we emulate `leave=True` behavior when closing
            # by printing a dummy newline and refreshing to force tqdm to print to a stale line
            leave=True,
            postfix=metrics,
            unit=unit.value,
        )

    def log_data(self, data: Dict[str, Any]):
        formatted_data = {k: format_log_data_value(v) for (k, v) in data.items() if k in self.keys_to_log}
        self.metrics.update(formatted_data)
        self.pbar.set_postfix(self.metrics)

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
            'metrics': self.metrics,
            'keys_to_log': self.keys_to_log,
            'n': pbar_state['n'],
            'unit': self.unit,
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
        log_to_console: Optional[bool] = None,
        console_log_level: Union[LogLevel, str, Callable[[State, LogLevel], bool]] = LogLevel.EPOCH,
        stream: Union[str, TextIO] = sys.stderr,
    ) -> None:

        self._show_pbar = progress_bar
        # The dummy pbar is to fix issues when streaming progress bars over k8s, where the progress bar in position 0
        # doesn't update until it is finished.
        # Need to have a dummy progress bar in position 0, so the "real" progress bars in positions 1 and 2 won't jump around
        self.dummy_pbar: Optional[_ProgressBar] = None
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
    def _current_pbar(self) -> Optional[_ProgressBar]:
        if self.eval_pbar is not None:
            return self.eval_pbar
        return self.train_pbar

    @property
    def show_pbar(self) -> bool:
        return self._show_pbar and dist.get_local_rank() == 0

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]) -> None:
        # log to progress bar
        if self._current_pbar:
            # Logging outside an epoch
            self._current_pbar.log_data(data)

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
        if self._current_pbar:
            # use tqdm.write to avoid interleaving
            self._current_pbar.pbar.write(log_str)
        else:
            # write directly to self.stream; no active progress bar
            print(log_str, file=self.stream, flush=True)

    def _build_pbar(self, state: State, is_train: bool) -> _ProgressBar:
        """Builds a pbar that tracks in the units of max_duration.

        Example:
            Samples     train  73% ||███████████████        | 293873/400000

        If epoch_style = True, then the pbar total will be the
        numbers of batches in the epoch, regardless of the max_duration units.
        This is often used to emit a pbar for each epoch, e.g.
            Epoch     0 train 100%|█████████████████████████| 29/29
            Epoch     1 train 100%|█████████████████████████| 29/29
        """
        # Always using position=1 to avoid jumping progress bars
        position = 1
        label = state.dataloader_label
        assert state.max_duration is not None, 'max_duration should be set'

        if state.max_duration.unit == TimeUnit.EPOCH:
            total = int(state.dataloader_len) if state.dataloader_len is not None else None
            timestamp_key = 'batch_in_epoch'

            unit = TimeUnit.BATCH
            n = state.timestamp.epoch.value
            if self.train_pbar is None and not is_train:
                # epochwise eval results refer to model from previous epoch (n-1)
                n -= 1
            desc = f'Epoch {n:3d}'
            if self.train_pbar is not None:
                # Evaluating mid-epoch
                batch_in_epoch = int(state.timestamp.batch_in_epoch)
                desc += f', Batch {batch_in_epoch:3d}: '
            else:
                desc += ': ' + ' ' * 11  # padding
        else:
            total = state.max_duration.value
            unit = state.max_duration.unit
            timestamp_key = unit.name.lower()
            desc = f'{unit.name.capitalize():<11}'
        
        desc += f'{label:15s}'

        return _ProgressBar(
            file=self.stream,
            total=total,
            position=position,
            keys_to_log=_IS_TRAIN_TO_KEYS_TO_LOG[is_train],
            bar_format=f'{desc} {{l_bar}}{{bar:25}}{{r_bar}}{{bar:-1b}}',
            unit=unit,
            metrics={},
            timestamp_key=timestamp_key,
        )

    def fit_start(self, state: State, logger: Logger) -> None:
        self.dummy_pbar = _ProgressBar(
            file=self.stream,
            position=0,
            total=2,
            metrics={},
            keys_to_log=[],
            bar_format='{bar:-1b}',
            unit=TimeUnit.DURATION,
            timestamp_key='',
        )
        self.dummy_pbar.update(1)

    def epoch_start(self, state: State, logger: Logger) -> None:
        if self.show_pbar and not self.train_pbar:
            self.train_pbar = self._build_pbar(state=state, is_train=True)

    def eval_start(self, state: State, logger: Logger) -> None:
        if self.show_pbar:
            self.eval_pbar = self._build_pbar(state, is_train=False)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.train_pbar:
            self.train_pbar.update_to_timestamp(state.timestamp)

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if self.eval_pbar:
            self.eval_pbar.update_to_timestamp(state.eval_timestamp)

    def epoch_end(self, state: State, logger: Logger) -> None:
        # only close the progress bar if its epoch_style
        # Otherwise, the same progress bar is used for all of training, so do not close it here
        assert state.max_duration is not None, 'max_duration should be set'
        if self.train_pbar and state.max_duration.unit == TimeUnit.EPOCH:
            self.train_pbar.close()
            self.train_pbar = None

    def fit_end(self, state: State, logger: Logger) -> None:
        # If the train pbar isn't closed (i.e. not epoch style), then it would still be open here
        if self.train_pbar:
            self.train_pbar.close()
            self.train_pbar = None
        if self.dummy_pbar:
            self.dummy_pbar.close()
            self.dummy_pbar = None

    def eval_end(self, state: State, logger: Logger) -> None:
        assert self.eval_pbar is not None
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
            self.train_pbar = _ProgressBar(file=self.stream, **state['train_pbar'])
            self.train_pbar.update(n=n)
        if state['eval_pbar']:
            n = state['train_pbar'].pop('n')
            self.eval_pbar = _ProgressBar(file=self.stream, **state['eval_pbar'])
            self.eval_pbar.update(n=n)
