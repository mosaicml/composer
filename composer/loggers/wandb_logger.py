# Copyright 2021 MosaicML. All Rights Reserved.

"""Log to Weights and Biases (https://wandb.ai/)"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import textwrap
import warnings
from typing import Any, Dict, Optional

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ["WandBLogger"]


class WandBLogger(LoggerDestination):
    """Log to Weights and Biases (https://wandb.ai/)

    Args:
        log_artifacts (bool, optional): Whether to log
            `artifacts <https://docs.wandb.ai/ref/python/artifact>`_ (Default: ``False``).
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
                 rank_zero_only: bool = True,
                 init_params: Optional[Dict[str, Any]] = None) -> None:
        try:
            import wandb
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group="wandb",
                                                conda_package="wandb",
                                                conda_channel="conda-forge") from e

        del wandb  # unused
        if log_artifacts and rank_zero_only:
            warnings.warn(
                textwrap.dedent("""\
                    When logging artifacts, `rank_zero_only` should be set to False.
                    Artifacts from other ranks will not be collected, leading to a loss of information required to
                    restore from checkpoints."""))
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self._log_artifacts = log_artifacts
        if init_params is None:
            init_params = {}
        self._init_params = init_params

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        import wandb
        del log_level  # unused
        if self._enabled:
            wandb.log(data, step=int(state.timer.batch))

    def state_dict(self) -> Dict[str, Any]:
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
        del state  # unused
        if "name" not in self._init_params:
            # Use the logger run name if the name is not set.
            self._init_params["name"] = logger.run_name

        if self._enabled:
            wandb.init(**self._init_params)

    def log_file_artifact(self, state: State, log_level: LogLevel, artifact_name: str, file_path: pathlib.Path, *,
                          overwrite: bool):
        del state, log_level, overwrite  # unused

        # replace all unsupported characters with periods
        # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
        new_artifact_name = re.sub(r'[^a-zA-Z0-9-_\.]', '.', artifact_name)
        if new_artifact_name != artifact_name:
            warnings.warn(("WandB permits only alpha-numeric, periods, hyphens, and underscores in artifact names. "
                           f"The artifact with name '{artifact_name}' will be stored as '{new_artifact_name}'."))

        if self._enabled and self._log_artifacts:
            import wandb
            extension = file_path.name.split(".")[-1]
            artifact = wandb.Artifact(name=new_artifact_name, type=extension)
            artifact.add_file(os.path.abspath(file_path))
            wandb.log_artifact(artifact)

    def post_close(self) -> None:
        import wandb

        # Cleaning up on post_close so all artifacts are uploaded
        if not self._enabled:
            return

        exc_tpe, exc_info, tb = sys.exc_info()

        if (exc_tpe, exc_info, tb) == (None, None, None):
            wandb.finish(0)
        else:
            # record there was an error
            wandb.finish(1)
