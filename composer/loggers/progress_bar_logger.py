# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Logs metrics to the console and show a progress bar."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, TextIO, Union

import tqdm.auto

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel, format_log_data_value
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist

__all__ = ["ProgressBarLogger"]

_IS_TRAIN_TO_KEYS_TO_LOG = {True: ['loss/train'], False: ['accuracy/val']}


@dataclass
class _ProgressBarLoggerInstanceState:
    total: Optional[int]
    description: str
    position: int
    keys_to_log: List[str]
    n: int
    epoch_metrics: Dict[str, Any]


class _ProgressBarLoggerInstance:

    def __init__(self, file: TextIO, state: _ProgressBarLoggerInstanceState) -> None:
        self.state = state
        self.pbar = tqdm.auto.tqdm(
            total=state.total,
            desc=state.description,
            position=state.position,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            file=file,
            dynamic_ncols=True,
        )
        self.pbar.set_postfix(state.epoch_metrics)

    def log_data(self, data: Dict[str, Any]):
        formatted_data = {k: format_log_data_value(v) for (k, v) in data.items() if k in self.state.keys_to_log}
        self.state.epoch_metrics.update(formatted_data)
        self.pbar.set_postfix(self.state.epoch_metrics)

    def update(self):
        self.pbar.update()
        self.state.n = self.pbar.n

    def close(self):
        self.pbar.close()

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self.state)


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
        self.show_pbar = progress_bar
        log_to_console = not progress_bar if log_to_console is None else log_to_console
        if not log_to_console:
            console_log_level = lambda state, ll: False
        self.pbars: Dict[bool, _ProgressBarLoggerInstance] = {}
        self.is_train: Optional[bool] = None

        if isinstance(console_log_level, str):
            console_log_level = LogLevel(console_log_level)
        if isinstance(console_log_level, LogLevel):

            def should_log(state: State, log_level: LogLevel, console_log_level: LogLevel = console_log_level):
                del state  # unused
                return log_level <= console_log_level

            self.should_log = should_log
        else:
            self.should_log = console_log_level
        if isinstance(stream, str):
            if stream.lower() == "stdout":
                stream = sys.stdout
            elif stream.lower() == "stderr":
                stream = sys.stderr
            else:
                raise ValueError("Invalid stream option: Should be 'stdout', 'stderr', or a TextIO-like object.")
        self.stream = stream

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]) -> None:
        if dist.get_local_rank() == 0 and self.is_train in self.pbars:
            # Logging outside an epoch
            assert self.is_train is not None
            self.pbars[self.is_train].log_data(data)

        if not self.should_log(state, log_level):
            return
        data_str = format_log_data_value(data)
        log_str = f'[{log_level.name}][batch={int(state.timestamp.batch)}]: {data_str}'

        if self.is_train in self.pbars:
            # use tqdm.write to avoid interleaving with a progress bar
            assert self.is_train is not None
            self.pbars[self.is_train].pbar.write(log_str)
        else:
            # write directly to self.stream; no active progress bar
            print(log_str, file=self.stream, flush=True)

    def _start(self, state: State):
        if dist.get_local_rank() != 0 or not self.show_pbar:
            return
        assert self.is_train is not None, "self.is_train should be set by the callback"
        assert state.dataloader_len is not None, "dataloader_len should be set when using tqdm"

        desc = f'Epoch {int(state.timestamp.epoch)}'
        position = 0 if self.is_train else 1
        if not self.is_train:
            desc += f", Batch {int(state.timestamp.batch)} (val)"
        self.pbars[self.is_train] = _ProgressBarLoggerInstance(
            file=self.stream,
            state=_ProgressBarLoggerInstanceState(
                total=int(state.dataloader_len),
                position=position,
                n=0,
                keys_to_log=_IS_TRAIN_TO_KEYS_TO_LOG[self.is_train],
                description=desc,
                epoch_metrics={},
            ),
        )

    def epoch_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if dist.get_local_rank() != 0:
            return
        self.is_train = True
        self._start(state)

    def eval_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if dist.get_local_rank() != 0:
            return
        self.is_train = False
        self._start(state)

    def _update(self):
        if dist.get_local_rank() != 0:
            return
        if self.is_train in self.pbars:
            assert self.is_train is not None
            self.pbars[self.is_train].update()

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_local_rank() != 0:
            return
        self._update()

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_local_rank() != 0:
            return
        self._update()

    def _end(self):
        if dist.get_local_rank() != 0:
            return
        if self.is_train in self.pbars:
            assert self.is_train is not None
            self.pbars[self.is_train].close()
            del self.pbars[self.is_train]
            self.is_train = None

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_local_rank() != 0:
            return
        self._end()

    def eval_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_local_rank() != 0:
            return
        self._end()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "pbars": {k: v.state_dict() for (k, v) in self.pbars.items()},
            "is_train": self.is_train,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.pbars = {}
        for is_train, pbar_state in state["pbars"].items():
            self.pbars[is_train] = _ProgressBarLoggerInstance(
                file=self.stream,
                state=_ProgressBarLoggerInstanceState(**pbar_state),
            )

        self.is_train = state["is_train"]
