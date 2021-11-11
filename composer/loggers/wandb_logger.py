# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import atexit
import os
import sys
from typing import Any

import wandb

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData
from composer.core.types import Logger, State, StateDict

from composer.utils.run_directory import get_run_directory  # isort:skip


class WandBLoggerBackend(RankZeroLoggerBackend):
    """Log to Weights and Biases (https://wandb.ai/)

    Args:
        log_artifacts_every_n_batches (int, optional): Interval at which to upload
            artifcats to wandb from the `run_directory`. On resnet50, a 22% regression
            was realized when logging and uploading artifacts, so it is recommended to
            do so infrequently. (default: ``100``)
        kwargs (Any): Parameters to pass into :meth:`wandb.init`.
    """

    def __init__(self, log_artifacts_every_n_batches: int = 100, **kwargs: Any) -> None:
        super().__init__()
        self.log_artifacts_every_n_batches = log_artifacts_every_n_batches
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

    def batch_end(self, state: State, logger: Logger) -> None:
        del logger  # unused
        # On resnet50, _log_artifacts() caused a 22% throughput degradation
        # wandb.log_artifact() is async according to the docs
        # (see https://docs.wandb.ai/guides/artifacts/api#2.-create-an-artifact)
        # so uploads will not block the training loop
        # slowdown is likely from extra I/O
        # Hence, logging every n batches instead of every batch
        if (state.step + 1) % self.log_artifacts_every_n_batches:
            self._log_artifacts()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._log_artifacts()

    def _log_artifacts(self):
        # Scan the run directory and upload artifacts to wandb
        run_directory = get_run_directory()
        if run_directory is not None:
            for subfile in os.listdir(run_directory):
                artifact = wandb.Artifact(name=subfile, type=subfile)
                full_path = os.path.join(run_directory, subfile)
                if os.path.isdir(full_path):
                    artifact.add_dir(full_path)
                else:
                    artifact.add_file(full_path)
                wandb.log_artifact(artifact)

    def _close_wandb(self) -> None:
        self._log_artifacts()

        exc_tpe, exc_info, tb = sys.exc_info()

        if (exc_tpe, exc_info, tb) == (None, None, None):
            wandb.finish(0)
        else:
            # record there was an error
            wandb.finish(1)
