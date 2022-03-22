# Copyright 2021 MosaicML. All Rights Reserved.

"""Logs metrics to a `TQDM <https://github.com/tqdm/tqdm>`_ progress bar displayed in the terminal."""

from __future__ import annotations

import collections.abc
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from tqdm import auto

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

    def __init__(self, state: _ProgressBarLoggerInstanceState) -> None:
        self.state = state
        self.pbar = auto.tqdm(total=state.total,
                              desc=state.description,
                              position=state.position,
                              bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
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
    """Logs metrics to a `TQDM <https://github.com/tqdm/tqdm>`_ progress bar displayed in the terminal.

    During training, the progress bar logs the batch and training loss.
    During validation, the progress bar logs the batch and validation accuracy.

    Example usage:
        .. testcode::

            from composer.loggers import ProgressBarLogger
            from composer.trainer import Trainer

            trainer = Trainer(
                ...,
                loggers=[ProgressBarLogger()]
            )

    Example output::

        Epoch 1: 100%|██████████| 64/64 [00:01<00:00, 53.17it/s, loss/train=2.3023]
        Epoch 1 (val): 100%|██████████| 20/20 [00:00<00:00, 100.96it/s, accuracy/val=0.0995]

    .. note::

        It is currently not possible to show additional metrics.
        Custom metrics for the TQDM progress bar will be supported in a future version.
    """

    def __init__(self) -> None:
        self.pbars: Dict[bool, _ProgressBarLoggerInstance] = {}
        self.is_train: Optional[bool] = None

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]) -> None:
        del state
        if dist.get_global_rank() == 0 and log_level <= LogLevel.BATCH and self.is_train in self.pbars:
            # Logging outside an epoch
            assert self.is_train is not None
            self.pbars[self.is_train].log_data(data)

    def _start(self, state: State):
        if dist.get_global_rank() != 0:
            return
        assert self.is_train is not None, "self.is_train should be set by the callback"
        if self.is_train:
            total_steps = state.steps_per_epoch
        else:
            total_steps = 0
            for evaluator in state.evaluators:
                dataloader_spec = evaluator.dataloader
                assert isinstance(dataloader_spec.dataloader, collections.abc.Sized)
                total_steps += len(dataloader_spec.dataloader)

        desc = f'Epoch {int(state.timer.epoch)}'
        position = 0 if self.is_train else 1
        if not self.is_train:
            desc += f", Batch {int(state.timer.batch)} (val)"
        self.pbars[self.is_train] = _ProgressBarLoggerInstance(
            _ProgressBarLoggerInstanceState(total=total_steps,
                                            position=position,
                                            n=0,
                                            keys_to_log=_IS_TRAIN_TO_KEYS_TO_LOG[self.is_train],
                                            description=desc,
                                            epoch_metrics={}))

    def epoch_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if dist.get_global_rank() != 0:
            return
        self.is_train = True
        self._start(state)

    def eval_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if dist.get_global_rank() != 0:
            return
        self.is_train = False
        self._start(state)

    def _update(self):
        if dist.get_global_rank() != 0:
            return
        if self.is_train in self.pbars:
            assert self.is_train is not None
            self.pbars[self.is_train].update()

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_global_rank() != 0:
            return
        self._update()

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_global_rank() != 0:
            return
        self._update()

    def _end(self):
        if dist.get_global_rank() != 0:
            return
        if self.is_train in self.pbars:
            assert self.is_train is not None
            self.pbars[self.is_train].close()
            del self.pbars[self.is_train]
            self.is_train = None

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_global_rank() != 0:
            return
        self._end()

    def eval_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if dist.get_global_rank() != 0:
            return
        self._end()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "pbars": {k: v.state_dict() for (k, v) in self.pbars.items()},
            "is_train": self.is_train,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.pbars = {
            k: _ProgressBarLoggerInstance(_ProgressBarLoggerInstanceState(**v)) for (k, v) in state["pbars"].items()
        }
        self.is_train = state["is_train"]
