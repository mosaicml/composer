# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `MLFlow <https://www.mlflow.org/docs/latest/index.html>."""

from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Optional, Union

from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

__all__ = ['MLFlowLogger']

DEFAULT_MLFLOW_EXPERIMENT_NAME = 'my-mlflow-experiment'


class MLFlowLogger(LoggerDestination):
    """Log to `MLFlow <https://www.mlflow.org/docs/latest/index.html>`_.

    Args:
        experiment_name: (str, optional): MLFLow experiment name. If not set it will be
            use the MLFLOW environment variable or a default value
        run_name: (str, optional): MLFlow run name. If not set it will be the same as the
            logger run name
        tracking_uri (str | pathlib.Path, optional): MLFlow tracking uri, the URI to the
            remote or local endpoint where logs are stored (If none it is set to MLFlow default)
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process
            (default: ``True``).
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[Union[str, pathlib.Path]] = None,
        rank_zero_only: bool = True,
    ) -> None:
        try:
            import mlflow
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='mlflow',
                                                conda_package='mlflow',
                                                conda_channel='conda-forge') from e
        del mlflow
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self.run_name = run_name
        self.experiment_name = experiment_name
        self._rank_zero_only = rank_zero_only
        self.tracking_uri = tracking_uri

    def _mlflow_set_experiment(self) -> None:
        import mlflow

        # Set experiment name defaults
        if self.experiment_name is not None:
            # Respect MLFlow environment variables before setting defaults
            env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
            if env_exp_id:
                # Experiment Id and name are mutually exclusive in MLFlow
                mlflow.set_experiment(experiment_id=env_exp_id)
                return

            # If the experiment name env var is not set, use the default
            self.experiment_name = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.name,
                                             DEFAULT_MLFLOW_EXPERIMENT_NAME)

        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _mlflow_start_run(self, state: State) -> None:
        import mlflow

        # Set run name defaults
        if self.run_name is None:
            # Respect MLFlow environment variables before setting defaults
            env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
            if env_run_id:
                # Run Id and name are mutually exclusive in MLFlow
                mlflow.start_run(run_id=env_run_id)
                return

            # If the run name env var is not set, use the logger run name.
            self.run_name = os.getenv(mlflow.environment_variables.MLFLOW_RUN_NAME.name, state.run_name)

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.run_name += f'-rank{dist.get_global_rank()}'

        mlflow.start_run(run_name=self.run_name)

    def init(self, state: State, logger: Logger) -> None:
        import mlflow
        del logger  # unused

        if self._enabled:
            if self.tracking_uri is not None:
                mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow_set_experiment()
            self._mlflow_start_run(state)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        import mlflow
        if self._enabled:
            # Convert all metrics to floats to placate mlflow.
            metrics = {k: float(v) for k, v in metrics.items()}
            mlflow.log_metrics(metrics=metrics, step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        import mlflow
        if self._enabled:
            mlflow.log_params(params=hyperparameters)

    def post_close(self):
        import mlflow
        if self._enabled:
            mlflow.end_run()
