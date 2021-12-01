# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import tqdm
import yaml

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData, TLogDataValue, format_log_data_value
from composer.core.state import State
from composer.core.types import StateDict

if TYPE_CHECKING:
    from composer.core.logging import Logger

_IS_TRAIN_TO_KEYS_TO_LOG = {True: ['loss/train'], False: ['accuracy/val']}


@dataclass
class _TQDMLoggerInstanceState:
    total: int
    epoch: int
    is_train: bool
    n: int
    epoch_metrics: Dict[str, TLogDataValue] = field(default_factory=dict)


class _TQDMLoggerInstance:

    def __init__(self,
                 total: int,
                 epoch: int,
                 is_train: bool,
                 n: int = 0,
                 epoch_metrics: Optional[Dict[str, TLogDataValue]] = None) -> None:
        self.state = _TQDMLoggerInstanceState(total=total,
                                              epoch=epoch,
                                              is_train=is_train,
                                              n=n,
                                              epoch_metrics=(epoch_metrics or {}))
        desc = f'Epoch {epoch + 1}{"" if is_train else " (val)"}'
        position = 0 if is_train else 1
        self.pbar = tqdm.tqdm(total=total,
                              desc=desc,
                              position=position,
                              initial=n,
                              bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        self.pbar.set_postfix(epoch_metrics)

    def log_metric(self, data: TLogData):
        formatted_data = {
            k: format_log_data_value(v) for (k, v) in data.items() if k in _IS_TRAIN_TO_KEYS_TO_LOG[self.state.is_train]
        }
        self.state.epoch_metrics.update(formatted_data)
        self.pbar.set_postfix(self.state.epoch_metrics)

    def update(self):
        self.pbar.update()
        self.state.n = self.pbar.n

    def close(self):
        self.pbar.close()

    def state_dict(self) -> StateDict:
        return asdict(self.state)


class TQDMLoggerBackend(RankZeroLoggerBackend):
    """Shows TQDM progress bars.

    During training, the progress bar logs the batch and training loss.
    During validation, the progress bar logs the batch and validation accuracy.

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

    def _will_log(self, state: State, log_level: LogLevel) -> bool:
        del state  # Unused
        return log_level <= LogLevel.BATCH

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        del epoch, step, log_level  # Unused
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
        assert self.is_train is not None, "self.is_train should be set by the callback"
        total_steps = len(state.train_dataloader) if self.is_train else len(state.eval_dataloader)
        self.pbars[self.is_train] = _TQDMLoggerInstance(total=total_steps, epoch=state.epoch, is_train=self.is_train)

    def epoch_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.is_train = True
        self._start(state)

    def eval_start(self, state: State, logger: Logger) -> None:
        del logger  # unused
        self.is_train = False
        self._start(state)

    def _update(self):
        if self.is_train in self.pbars:
            assert self.is_train is not None
            self.pbars[self.is_train].update()

    def after_backward(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._update()

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._update()

    def _end(self):
        if self.is_train in self.pbars:
            assert self.is_train is not None
            self.pbars[self.is_train].close()
            del self.pbars[self.is_train]
            self.is_train = None

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._end()

    def eval_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._end()

    def state_dict(self) -> StateDict:
        return {
            "pbars": {k: v.state_dict() for (k, v) in self.pbars.items()},
            "is_train": self.is_train,
        }

    def load_state_dict(self, state: StateDict) -> None:
        self.pbars = {k: _TQDMLoggerInstance(**v) for (k, v) in state["pbars"].items()}
        self.is_train = state["is_train"]
