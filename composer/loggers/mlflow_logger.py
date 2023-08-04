# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `MLFlow <https://www.mlflow.org/docs/latest/index.html>."""

from __future__ import annotations

import os
import pathlib
import time
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
            Trainer run name
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
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self.run_name = run_name
        self.experiment_name = experiment_name
        self._rank_zero_only = rank_zero_only
        self.tracking_uri = str(tracking_uri or mlflow.get_tracking_uri())
        self.metrics_batch_number = 0
        self._last_flush_time = time.time()
        del mlflow

    def init(self, state: State, logger: Logger) -> None:
        import mlflow
        from mlflow import MlflowClient
        from mlflow.utils.autologging_utils import MlflowAutologgingQueueingClient

        del logger  # unused

        if self.experiment_name is None:
            self.experiment_name = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.name,
                                             DEFAULT_MLFLOW_EXPERIMENT_NAME)

        if self.run_name is None:
            self.run_name = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.run_name += f'-rank{dist.get_global_rank()}'

        if self._enabled:
            self._mlflow_client = MlflowAutologgingQueueingClient(self.tracking_uri)

            # set experiment
            # NB: we use MlflowClient for experiment creation because
            # MlflowAutologgingQueueingClient doesn't support it
            env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
            if env_exp_id is not None:
                self._experiment_id = env_exp_id
            elif experiment := MlflowClient(self.tracking_uri).get_experiment_by_name(name=self.experiment_name):
                self._experiment_id = experiment.experiment_id
            else:
                self._experiment_id = MlflowClient(self.tracking_uri).create_experiment(name=self.experiment_name)

            # start run
            env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
            if env_run_id is not None:
                self._run_id = env_run_id
            else:
                self._run_id = MlflowClient(self.tracking_uri).create_run(experiment_id=self._experiment_id, run_name=self.run_name).info.run_id

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            metrics = {k: float(v) for k, v in metrics.items()}
            self._mlflow_client.log_metrics(
                run_id=self._run_id,
                metrics=metrics,
                step=step,
            )
            time_since_flush = (time.time() - self._last_flush_time)
            if time_since_flush >= 10:
                self._mlflow_client.flush(synchronous=False)
                self._last_flush_time = time.time()

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            self._mlflow_client.log_params(
                run_id=self._run_id,
                params=hyperparameters,
            )
            self._mlflow_client.flush(synchronous=False)

    def post_close(self):
        if self._enabled:
            self._mlflow_client.set_terminated(self._run_id)
            self._mlflow_client.flush(synchronous=True)
