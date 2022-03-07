# Copyright 2021 MosaicML. All Rights Reserved.

"""Log to Weights and Biases (https://wandb.ai/)"""

from __future__ import annotations

import datetime
import os
import sys
import textwrap
import warnings
from typing import Any, Dict, Optional

from composer.core.logging import LoggerCallback, LogLevel, TLogData
from composer.core.time import Timestamp
from composer.core.types import Logger, State, StateDict
from composer.utils import dist, run_directory

__all__ = ["WandBLogger"]


class WandBLogger(LoggerCallback):
    """Log to Weights and Biases (https://wandb.ai/)

    Args:
        log_artifacts (bool, optional): Whether to log
            `artifacts <https://docs.wandb.ai/ref/python/artifact>`_ (Default: ``False``).
        log_artifacts_every_n_batches (int, optional): Interval at which to upload
            `artifacts <https://docs.wandb.ai/ref/python/artifact>`_ to wandb from the
            :mod:`~.composer.utils.run_directory`. Logging very frequently (e.g. on every
            batch) can substantially slow down training, so we recommend doing so
            infrequently. Only applicable when ``log_artifacts`` is True (default:
            ``100``).
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            When logging `artifacts <https://docs.wandb.ai/ref/python/artifact>`_, it is
            highly recommended to log on all ranks.  Artifacts from ranks ≥1 will not be
            stored, which may discard pertinent information. For example, when using
            Deepspeed ZeRO, it would be impossible to restore from checkpoints without
            artifacts from all ranks (default: ``False``).
        init_params (Dict[str, Any], optional): Parameters to pass into
            ``wandb.init`` (see
            `WandB documentation <https://docs.wandb.ai/ref/python/init>`_).
    """

    def __init__(self,
                 log_artifacts: bool = False,
                 log_artifacts_every_n_batches: int = 100,
                 rank_zero_only: bool = True,
                 init_params: Optional[Dict[str, Any]] = None) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                textwrap.dedent("""\
                Composer was installed without WandB support. To use WandB with Composer, run `pip install mosaicml[wandb]`
                if using pip or `conda install -c conda-forge wandb` if using Anaconda.""")) from e
        del wandb  # unused
        if log_artifacts and rank_zero_only:
            warnings.warn(
                textwrap.dedent("""\
                    When logging artifacts, `rank_zero_only` should be set to False.
                    Artifacts from other ranks will not be collected, leading to a loss of information required to
                    restore from checkpoints."""))
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self._log_artifacts = log_artifacts
        self._log_artifacts_every_n_batches = log_artifacts_every_n_batches
        self._last_upload_timestamp = datetime.datetime.fromtimestamp(0)
        if init_params is None:
            init_params = {}
        self._init_params = init_params

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
        import wandb
        del log_level  # unused
        if self._enabled:
            wandb.log(data, step=int(timestamp.batch))

    def state_dict(self) -> StateDict:
        import wandb

        # Storing these fields in the state dict to support run resuming in the future.
        if self._enabled:
            if wandb.run is None:
                raise ValueError("wandb must be initialized before serialization.")
            return {
                "name": wandb.run.name,
                "project": wandb.run.project,
                "entity": wandb.run.entity,
                "id": wandb.run.id,
                "group": wandb.run.group
            }
        else:
            return {}

    def init(self, state: State, logger: Logger) -> None:
        import wandb
        del state, logger  # unused
        if self._enabled:
            wandb.init(**self._init_params)

    def batch_end(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if self._enabled and self._log_artifacts and int(
                state.timer.batch_in_epoch) % self._log_artifacts_every_n_batches == 0:
            self._upload_artifacts()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        if self._enabled and self._log_artifacts:
            self._upload_artifacts()

    def _upload_artifacts(self):
        import wandb

        # Scan the run directory and upload artifacts to wandb
        # On resnet50, _log_artifacts() caused a 22% throughput degradation
        # wandb.log_artifact() is async according to the docs
        # (see https://docs.wandb.ai/guides/artifacts/api#2.-create-an-artifact)
        # so uploads will not block the training loop
        # slowdown is likely from extra I/O of scanning the directory and/or
        # scheduling uploads
        modified_files = run_directory.get_modified_files(self._last_upload_timestamp)
        for modified_file in modified_files:
            file_type = modified_file.split(".")[-1]
            relpath = os.path.relpath(modified_file, run_directory.get_run_directory())
            relpath = f"rank_{dist.get_global_rank()}-" + relpath.replace("/", "-")
            artifact = wandb.Artifact(name=relpath, type=file_type)
            artifact.add_file(os.path.abspath(modified_file))
            wandb.log_artifact(artifact)
        self._last_upload_timestamp = run_directory.get_run_directory_timestamp()

    def post_close(self) -> None:
        import wandb

        # Cleaning up on post_close so all artifacts are uploaded
        if not self._enabled:
            return

        if self._log_artifacts:
            self._upload_artifacts()

        exc_tpe, exc_info, tb = sys.exc_info()

        if (exc_tpe, exc_info, tb) == (None, None, None):
            wandb.finish(0)
        else:
            # record there was an error
            wandb.finish(1)
