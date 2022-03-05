# Copyright 2021 MosaicML. All Rights Reserved.

"""Logs metrics to a `TQDM <https://github.com/tqdm/tqdm>`_ progress bar displayed in the terminal."""

from __future__ import annotations

import collections.abc
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml
from tqdm import auto

from composer.core.logging import LogLevel, TLogData, TLogDataValue, format_log_data_value
from composer.core.logging.base_backend import LoggerCallback
from composer.core.state import State
from composer.core.time import Timestamp
from composer.core.types import StateDict
from composer.utils import dist

if TYPE_CHECKING:
    from composer.core.logging import Logger

__all__ = ["TQDMLogger"]

_IS_TRAIN_TO_KEYS_TO_LOG = {True: ['loss/train'], False: ['accuracy/val']}


@dataclass
class _TQDMLoggerInstanceState:
    total: Optional[int]
    description: str
    position: int
    keys_to_log: List[str]
    n: int
    epoch_metrics: Dict[str, TLogDataValue]


class _TQDMLoggerInstance:

    def __init__(self, state: _TQDMLoggerInstanceState) -> None:
        self.state = state
        self.pbar = auto.tqdm(total=state.total,
                              desc=state.description,
                              position=state.position,
                              bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        self.pbar.set_postfix(state.epoch_metrics)

    def log_metric(self, data: TLogData):
        formatted_data = {k: format_log_data_value(v) for (k, v) in data.items() if k in self.state.keys_to_log}
        self.state.epoch_metrics.update(formatted_data)
        self.pbar.set_postfix(self.state.epoch_metrics)

    def update(self):
        self.pbar.update()
        self.state.n = self.pbar.n

    def close(self):
        self.pbar.close()

    def state_dict(self) -> StateDict:
        return asdict(self.state)


class TQDMLogger(LoggerCallback):
    """Logs metrics to a `TQDM <https://github.com/tqdm/tqdm>`_ progress bar displayed in the terminal.

    During training, the progress bar logs the batch and training loss.
    During validation, the progress bar logs the batch and validation accuracy.

    Example usage:
        .. testcode::

            from composer.loggers import TQDMLogger
            from composer.trainer import Trainer
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                optimizers=[optimizer],
                loggers=[TQDMLogger()]
            )

    Example output::

        Epoch 1: 100%|██████████| 64/64 [00:01<00:00, 53.17it/s, loss/train=2.3023]
        Epoch 1 (val): 100%|██████████| 20/20 [00:00<00:00, 100.96it/s, accuracy/val=0.0995]

    .. note::

        It is currently not possible to show additional metrics.
        Custom metrics for the TQDM progress bar will be supported in a future version.

    Args:
        config (dict or None, optional):
            Trainer configuration. If provided, it is printed to the terminal as YAML.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.pbars: Dict[bool, _TQDMLoggerInstance] = {}
        self.is_train: Optional[bool] = None
        self.config = config

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        del state  # Unused
        return dist.get_global_rank() == 0 and log_level <= LogLevel.BATCH

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData) -> None:
        del timestamp, log_level  # Unused
        if self.is_train in self.pbars:
            # Logging outside an epoch
            assert self.is_train is not None
            self.pbars[self.is_train].log_metric(data)

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self.config is not None:
            print("Config")
            print("-" * 30)
            yaml.safe_dump(self.config, stream=sys.stdout)
            print("-" * 30)
            print()

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
        self.pbars[self.is_train] = _TQDMLoggerInstance(
            _TQDMLoggerInstanceState(total=total_steps,
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

    def state_dict(self) -> StateDict:
        return {
            "pbars": {k: v.state_dict() for (k, v) in self.pbars.items()},
            "is_train": self.is_train,
        }

    def load_state_dict(self, state: StateDict) -> None:
        self.pbars = {k: _TQDMLoggerInstance(_TQDMLoggerInstanceState(**v)) for (k, v) in state["pbars"].items()}
        self.is_train = state["is_train"]
