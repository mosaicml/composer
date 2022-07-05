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
        unit: TimeUnit = TimeUnit.EPOCH,
        epoch_style: bool = False,
    ) -> None:
        self.keys_to_log = keys_to_log
        self.metrics = metrics
        self.position = position
        self.unit = unit
        self.epoch_style = epoch_style
        self.pbar = tqdm.auto.tqdm(
            total=total,
            position=position,
            bar_format=bar_format,
            file=file,
            dynamic_ncols=True,
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
        if self.epoch_style:
            n = int(timestamp.batch_in_epoch)
        else:
            n = int(timestamp.get(self.unit))

        n = n - self.pbar.n
        self.pbar.update(int(n))

    def close(self):
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
            'epoch_style': self.epoch_style
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
        self.train_pbar: Optional[_ProgressBar] = None
        self.eval_pbar: Optional[_ProgressBar] = None
        self.is_train: Optional[bool] = None

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
        if self.is_train is None:
            return None
        return self.train_pbar if self.is_train else self.eval_pbar

    @_current_pbar.setter
    def _current_pbar(self, pbar: Optional[_ProgressBar]):
        assert self.is_train is not None, 'Cannot set pbar if self.is_train is not set.'

        if self.is_train:
            self.train_pbar = pbar
        else:
            self.eval_pbar = pbar

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

    def _build_pbar(self, state: State, is_train: bool, epoch_style: bool = False) -> _ProgressBar:
        """Builds a pbar that tracks in the units of max_duration.

        Example:
            Samples     train  73% ||███████████████        | 293873/400000

        If epoch_style = True, then the pbar total will be the
        numbers of batches in the epoch, regardless of the max_duration units.
        This is often used to emit a pbar for each epoch, e.g.
            Epoch     0 train 100%|█████████████████████████| 29/29
            Epoch     1 train 100%|█████████████████████████| 29/29
        """
        position = 0 if is_train else 1
        split = 'train' if is_train else 'val'

        assert self.is_train is not None
        if epoch_style:
            total = int(state.dataloader_len) if state.dataloader_len is not None else None

            # handle when # batches is less than an epoch
            if total is not None and state.max_duration is not None \
                and state.max_duration.unit == TimeUnit.BATCH:
                total = min(total, state.max_duration.value)

            unit = TimeUnit.BATCH
            n = state.timestamp.epoch.value
            if not is_train:
                # eval results refer to model from previous epoch (n-1)
                n = max(0, n - 1)
            desc = f'Epoch {n:5d} {split:5s}'
        else:
            assert state.max_duration is not None, 'max_duration should be set'

            total = state.max_duration.value
            unit = state.max_duration.unit
            desc = f'{unit.name.capitalize():<11} {split:5s}'

        return _ProgressBar(
            file=self.stream,
            total=total,
            position=position,
            keys_to_log=_IS_TRAIN_TO_KEYS_TO_LOG[self.is_train],
            bar_format=f'{desc} {{l_bar}}{{bar:25}}{{r_bar}}{{bar:-1b}}',
            unit=unit,
            metrics={},
            epoch_style=epoch_style,
        )

    def _close(self):
        if self._current_pbar:
            self._current_pbar.close()
            self._current_pbar = None
            self.is_train = None

    def epoch_start(self, state: State, logger: Logger) -> None:
        self.is_train = True
        assert state.max_duration is not None, 'max_duration should be set'
        epoch_style = state.max_duration.unit == TimeUnit.EPOCH
        if self.show_pbar and not self.train_pbar:
            self.train_pbar = self._build_pbar(state=state, is_train=True, epoch_style=epoch_style)

    def eval_start(self, state: State, logger: Logger) -> None:
        self.is_train = False
        if self.show_pbar:
            self.eval_pbar = self._build_pbar(state, is_train=False, epoch_style=True)

    def batch_end(self, state: State, logger: Logger) -> None:
        self.is_train = True
        if self.train_pbar:
            self.train_pbar.update_to_timestamp(state.timestamp)

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if self.eval_pbar:
            self.eval_pbar.update_to_timestamp(state.eval_timestamp)

    def epoch_end(self, state: State, logger: Logger) -> None:
        # only close the progress bar if its epoch_style
        if self.train_pbar and self.train_pbar.epoch_style:
            self._close()

    def eval_end(self, state: State, logger: Logger) -> None:
        self._close()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'train_pbar': self.train_pbar.state_dict() if self.train_pbar else None,
            'eval_pbar': self.eval_pbar.state_dict() if self.eval_pbar else None,
            'is_train': self.is_train,
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

        self.is_train = state['is_train']
