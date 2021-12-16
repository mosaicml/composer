# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from composer.core.logging import BaseLoggerBackend, LogLevel, TLogData
from composer.core.types import Logger, State, StateDict
from composer.utils import ddp, run_directory

import wandb  # isort: skip


class WandBLoggerBackend(BaseLoggerBackend):
    """Log to Weights and Biases (https://wandb.ai/)

    Args:
        log_artifacts (bool, optional): Whether to log artifacts (default: ``False``)
        log_artifacts_every_n_batches (int, optional): Interval at which to upload
            artifcats to wandb from the `run_directory`. On resnet50, a 22% regression
            was realized when logging and uploading artifacts, so it is recommended to
            do so infrequently. Only applicable when `log_artifacts` is True
            (default: ``100``)
        init_params (Dict[str, Any], optional): Parameters to pass into :meth:`wandb.init`.
    """

    def __init__(self,
                 log_artifacts: bool = False,
                 log_artifacts_every_n_batches: int = 100,
                 init_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self._log_artifacts = log_artifacts
        self._log_artifacts_every_n_batches = log_artifacts_every_n_batches
        self._last_upload_timestamp = 0.0
        if init_params is None:
            init_params = {}
        self._init_params = init_params

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        del epoch, log_level  # unused
        if ddp.get_local_rank() == 0:
            wandb.log(data, step=step)

    def state_dict(self) -> StateDict:
        # Storing these fields in the state dict to support run resuming in the future.
        if ddp.get_local_rank() != 0:
            raise RuntimeError("WandB can only be checkpointed on rank 0")
        return {"name": wandb.run.name, "project": wandb.run.project, "entity": wandb.run.entity, "id": wandb.run.id}

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if ddp.get_local_rank() == 0:
            wandb.init(**self._init_params)

    def batch_end(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if self._log_artifacts and (state.step + 1) % self._log_artifacts_every_n_batches == 0:
            self._upload_artifacts()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self._log_artifacts:
            self._upload_artifacts()

    def training_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self._log_artifacts:
            self._upload_artifacts()

    def _upload_artifacts(self):
        # Scan the run directory and upload artifacts to wandb
        # On resnet50, _log_artifacts() caused a 22% throughput degradation
        # wandb.log_artifact() is async according to the docs
        # (see https://docs.wandb.ai/guides/artifacts/api#2.-create-an-artifact)
        # so uploads will not block the training loop
        # slowdown is likely from extra I/O of scanning the directory and/or
        # scheduling uploads
        run_directory_name = run_directory.get_run_directory()
        if run_directory_name is None:
            return
        # barrier that every process has reached this point and that
        # previous callbacks are finished writing to the run directory
        ddp.barrier()
        if ddp.get_local_rank() == 0:
            modified_files = run_directory.get_modified_files(self._last_upload_timestamp)
            for modified_file in modified_files:
                file_type = modified_file.split(".")[-1]
                relpath = os.path.relpath(modified_file, run_directory_name)
                relpath = relpath.replace("/", "-")
                artifact = wandb.Artifact(name=relpath, type=file_type)
                artifact.add_file(os.path.abspath(modified_file))
                wandb.log_artifact(artifact)
            self._last_upload_timestamp = run_directory.get_run_directory_timestamp()
        # barrier to ensure that other processes do not continue to other callbacks
        # that could start writing to the run directory
        ddp.barrier()

    def post_close(self) -> None:
        # Cleaning up on post_close so all artifacts are uploaded
        if self._log_artifacts:
            self._upload_artifacts()

        exc_tpe, exc_info, tb = sys.exc_info()

        if ddp.get_local_rank() == 0:
            if (exc_tpe, exc_info, tb) == (None, None, None):
                wandb.finish(0)
            else:
                # record there was an error
                wandb.finish(1)
