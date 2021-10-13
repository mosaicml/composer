# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import atexit
import sys
from typing import Any

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData
from composer.core.types import Logger, State, StateDict

import wandb  # isort:skip


class WandBLoggerBackend(RankZeroLoggerBackend):
    """Log to Weights and Biases (https://wandb.ai/)

    Args:
        kwargs (Any): Parameters to pass into :meth:`wandb.init`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._init_params = kwargs

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        del epoch, log_level  # unused
        wandb.log(data, step=step)

    def _training_start(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        wandb.init(**self._init_params)
        atexit.register(self._close_wandb)

    def state_dict(self) -> StateDict:
        # Storing these fields in the state dict to support run resuming in the future.
        return {"name": wandb.run.name, "project": wandb.run.project, "entity": wandb.run.entity, "id": wandb.run.id}

    def _close_wandb(self) -> None:
        # wandb.finish isn't automatically called during ddp-spawn, so manually calling it on rank 0
        exc_tpe, exc_info, tb = sys.exc_info()
        if (exc_tpe, exc_info, tb) == (None, None, None):
            wandb.finish(0)
        else:
            # record there was an error
            wandb.finish(1)
