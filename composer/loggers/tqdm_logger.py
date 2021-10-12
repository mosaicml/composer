# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import yaml
from tqdm import tqdm

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData, TLogDataValue, format_log_data_value
from composer.core.state import State
from composer.core.types import StateDict

if TYPE_CHECKING:
    from composer.core.logging import Logger


@dataclass
class TQDMLoggerInstanceState:
    total: int
    epoch: int
    val: bool
    n: int
    keys_to_log: Sequence[str]
    epoch_metrics: Dict[str, TLogDataValue] = field(default_factory=dict)


class TQDMLoggerInstance:

    def __init__(self,
                 total: int,
                 epoch: int,
                 val: bool,
                 keys_to_log: Sequence[str],
                 n: int = 0,
                 epoch_metrics: Optional[Dict[str, TLogDataValue]] = None) -> None:
        self.state = TQDMLoggerInstanceState(total=total,
                                             epoch=epoch,
                                             val=val,
                                             n=n,
                                             keys_to_log=keys_to_log,
                                             epoch_metrics=(epoch_metrics or {}))
        desc = f'Epoch {epoch + 1}{" (val)" if val else ""}'
        position = 1 if val else 0
        self.pbar = tqdm(total=total,
                         desc=desc,
                         position=position,
                         initial=n,
                         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
        self.pbar.set_postfix(epoch_metrics)

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


class TQDMLoggerBackend(RankZeroLoggerBackend):

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.pbar_train: Optional[TQDMLoggerInstance] = None
        self.pbar_val: Optional[TQDMLoggerInstance] = None
        self.is_validating = False
        self.config = config

    def _will_log(self, state: State, log_level: LogLevel) -> bool:
        return log_level <= LogLevel.BATCH

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData) -> None:
        pbar = self.pbar_val if self.is_validating else self.pbar_train
        if pbar is None:
            # Logging outside an epoch
            return
        pbar.log_metric(data)

    def _training_start(self, state: State, logger: Logger) -> None:
        if self.config is not None:
            print("Config")
            print("-" * 30)
            yaml.safe_dump(self.config, stream=sys.stdout)
            print("-" * 30)
            print()

    def epoch_start(self, state: State, logger: Logger) -> None:
        assert self.pbar_train is None
        self.pbar_train = TQDMLoggerInstance(total=state.steps_per_epoch,
                                             epoch=state.epoch,
                                             val=False,
                                             keys_to_log=["loss/train"])

    def after_backward(self, state: State, logger: Logger) -> None:
        assert self.pbar_train is not None
        self.pbar_train.update()

    def epoch_end(self, state: State, logger: Logger) -> None:
        assert self.pbar_train is not None
        self.pbar_train.close()
        self.pbar_train = None

    def eval_start(self, state: State, logger: Logger) -> None:
        assert self.pbar_val is None
        assert state.eval_dataloader is not None
        self.pbar_val = TQDMLoggerInstance(total=len(state.eval_dataloader),
                                           epoch=state.epoch,
                                           val=True,
                                           keys_to_log=["accuracy/val"])
        self.is_validating = True

    def eval_after_forward(self, state: State, logger: Logger) -> None:
        assert self.pbar_val is not None
        self.pbar_val.update()

    def eval_end(self, state: State, logger: Logger) -> None:
        assert self.pbar_val is not None
        self.pbar_val.close()
        self.pbar_val = None
        self.is_validating = False

    def state_dict(self) -> StateDict:
        state = {"is_validating": self.is_validating}
        if self.pbar_train:
            state["pbar_train"] = self.pbar_train.state_dict()
        if self.pbar_val:
            state["pbar_val"] = self.pbar_val.state_dict()
        return state

    def load_state_dict(self, state: StateDict) -> None:
        self.is_validating = state["is_validating"]
        if "pbar_train" in state:
            self.pbar_train = TQDMLoggerInstance(**state["pbar_train"])
        if "pbar_val" in state:
            self.pbar_val = TQDMLoggerInstance(**state["pbar"])
