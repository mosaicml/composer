# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to the console and show a progress bar."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TextIO, Union

import tqdm.auto
import yaml

from composer.core.time import TimeUnit
from composer.loggers.logger import Logger, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist, is_notebook

if TYPE_CHECKING:
    from composer.core import State, Timestamp

__all__ = ['ProgressBarLogger']

_IS_TRAIN_TO_KEYS_TO_LOG = {True: ['loss/train'], False: ['metrics/eval/Accuracy']}


class _ProgressBar:

    def __init__(
        self,
        total: Optional[int],
        position: Optional[int],
        bar_format: str,
        file: TextIO,
        metrics: Dict[str, Any],
        keys_to_log: List[str],
        timestamp_key: str,
        unit: str = 'it',
    ) -> None:
        self.keys_to_log = keys_to_log
        self.metrics = metrics
        self.position = position
        self.timestamp_key = timestamp_key
        self.file = file
        is_atty = is_notebook() or os.isatty(self.file.fileno())
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
            # If in a notebook, then always set leave=True, as otherwise jupyter would remote the progress bars
            leave=True if is_notebook() else not is_atty,
            postfix=metrics,
            unit=unit,
        )

    def log_data(self, data: Dict[str, Any]):
        formatted_data = {}
        for (k, v) in data.items():
            # Check if any substring of the key matches the keys to log
            if any(key_to_log in k for key_to_log in self.keys_to_log):
                formatted_data[k] = format_log_data_value(v)

        self.metrics.update(formatted_data)
        self.pbar.set_postfix(self.metrics)

    def update(self, n=1):
        self.pbar.update(n=n)

    def update_to_timestamp(self, timestamp: Timestamp):
        n = int(getattr(timestamp, self.timestamp_key))
        n = n - self.pbar.n
        self.update(int(n))

    def close(self):
        if is_notebook():
            # If in a notebook, always refresh before closing, so the
            # finished progress is displayed
            self.pbar.refresh()
        else:
            if self.position != 0:
                # Force a (potentially hidden) progress bar to re-render itself
                # Don't render the dummy pbar (at position 0), since that will clear a real pbar (at position 1)
                self.pbar.refresh()
            # Create a newline that will not be erased by leave=False. This allows for the finished pbar to be cached in the terminal
            # This emulates `leave=True` without progress bar jumping
            if not self.file.closed:
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
            'timestamp_key': self.timestamp_key,
        }


class ProgressBarLogger(LoggerDestination):
    """Log metrics to the console and optionally show a progress bar.

    .. note::

        This logger is automatically instantiated by the trainer via the ``progress_bar``,
        and ``console_stream`` options. This logger does not need to be created manually.

    `TQDM <https://github.com/tqdm/tqdm>`_ is used to display progress bars.

    During training, the progress bar logs the batch and training loss.
    During validation, the progress bar logs the batch and validation accuracy.

    Example progress bar output::

        Epoch 1: 100%|██████████| 64/64 [00:01<00:00, 53.17it/s, loss/train=2.3023]
        Epoch 1 (val): 100%|██████████| 20/20 [00:00<00:00, 100.96it/s, accuracy/val=0.0995]

    Args:
        stream (str | TextIO, optional): The console stream to use. If a string, it can either be ``'stdout'`` or
            ``'stderr'``. (default: :attr:`sys.stderr`)
        log_traces (bool): Whether to log traces or not. (default: ``False``)
    """

    def __init__(
        self,
        stream: Union[str, TextIO] = sys.stderr,
        log_traces: bool = False,
    ) -> None:

        # The dummy pbar is to fix issues when streaming progress bars over k8s, where the progress bar in position 0
        # doesn't update until it is finished.
        # Need to have a dummy progress bar in position 0, so the "real" progress bars in position 1 doesn't jump around
        self.dummy_pbar: Optional[_ProgressBar] = None
        self.train_pbar: Optional[_ProgressBar] = None
        self.eval_pbar: Optional[_ProgressBar] = None

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
        self.state: Optional[State] = None
        self.hparams: Dict[str, Any] = {}
        self.hparams_already_logged_to_console: bool = False

    @property
    def show_pbar(self) -> bool:
        return dist.get_local_rank() == 0

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        # Lazy logging of hyperparameters.
        self.hparams.update(hyperparameters)

    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            self._log_to_console('*' * 30)
            self._log_to_console('Config:')
            self._log_to_console(yaml.dump(self.hparams))
            self._log_to_console('*' * 30)
        self.hparams_already_logged_to_console = True

    def log_traces(self, traces: Dict[str, Any]):
        if self.should_log_traces:
            for trace_name, trace in traces.items():
                trace_str = format_log_data_value(trace)
                self._log_to_console(f'[trace]: {trace_name}:' + trace_str + '\n')

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for metric_name, metric_value in metrics.items():
            # Only log metrics and losses to pbar.
            if 'metric' in metric_name or 'loss' in metric_name:
                self.log_to_pbar(data={metric_name: metric_value})

    def log_to_pbar(self, data: Dict[str, Any]):
        # log to progress bar
        current_pbar = self.eval_pbar if self.eval_pbar is not None else self.train_pbar
        if current_pbar:
            # Logging outside an epoch
            current_pbar.log_data(data)

    def _log_to_console(self, log_str: str):
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
        # In jupyter notebooks, no need for the dummy pbar, so use the default position
        position = None if is_notebook() else 1
        desc = f'{state.dataloader_label:15}'
        max_duration_unit = None if state.max_duration is None else state.max_duration.unit

        if max_duration_unit == TimeUnit.EPOCH or max_duration_unit is None:
            total = int(state.dataloader_len) if state.dataloader_len is not None else None
            timestamp_key = 'batch_in_epoch'

            unit = TimeUnit.BATCH
            n = state.timestamp.epoch.value
            if self.train_pbar is None and not is_train:
                # epochwise eval results refer to model from previous epoch (n-1)
                n = n - 1 if n > 0 else 0
            if self.train_pbar is None:
                desc += f'Epoch {n:3}'
            else:
                # For evaluation mid-epoch, show the total batch count
                desc += f'Batch {int(state.timestamp.batch):3}'
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
            keys_to_log=_IS_TRAIN_TO_KEYS_TO_LOG[is_train],
            # In a notebook, the `bar_format` should not include the {bar}, as otherwise
            # it would appear twice.
            bar_format=desc + ' {l_bar}' + ('' if is_notebook() else '{bar:25}') + '{r_bar}{bar:-1b}',
            unit=unit.value.lower(),
            metrics={},
            timestamp_key=timestamp_key,
        )

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if not is_notebook():
            # Notebooks don't need the dummy progress bar; otherwise, it would be visible.
            self.dummy_pbar = _ProgressBar(
                file=self.stream,
                position=0,
                total=1,
                metrics={},
                keys_to_log=[],
                bar_format='{bar:-1b}',
                timestamp_key='',
            )
        self.state = state

    def fit_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self._log_hparams_to_console()

    def predict_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self._log_hparams_to_console()

    def epoch_start(self, state: State, logger: Logger) -> None:
        if self.show_pbar and not self.train_pbar:
            self.train_pbar = self._build_pbar(state=state, is_train=True)

    def eval_start(self, state: State, logger: Logger) -> None:
        if not self.hparams_already_logged_to_console:
            self._log_hparams_to_console()
        if self.show_pbar:
            self.eval_pbar = self._build_pbar(state, is_train=False)

    def batch_end(self, state: State, logger: Logger) -> None:
        if self.train_pbar:
            self.train_pbar.update_to_timestamp(state.timestamp)

    def eval_batch_end(self, state: State, logger: Logger) -> None:
        if self.eval_pbar:
            self.eval_pbar.update_to_timestamp(state.eval_timestamp)

    def epoch_end(self, state: State, logger: Logger) -> None:
        # Only close progress bars at epoch end if the duration is in epochs, since
        # a new pbar will be created for each epoch
        # If the duration is in other units, then one progress bar will be used for all of training.
        assert state.max_duration is not None, 'max_duration should be set'
        if self.train_pbar and state.max_duration.unit == TimeUnit.EPOCH:
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
