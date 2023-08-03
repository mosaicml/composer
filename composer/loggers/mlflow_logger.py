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

from mlflow.utils.autologging_utils import MlflowAutologgingQueueingClient

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
        del mlflow
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self.run_name = run_name
        self.experiment_name = experiment_name
        self._rank_zero_only = rank_zero_only
        self.tracking_uri = tracking_uri
        self.metrics_batch_number = 0
        self._last_flush_time = time.time()

    def init(self, state: State, logger: Logger) -> None:
        import mlflow
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
            self._mlflow_client = MlflowAutologgingQueuingClient(self.tracking_uri)

            # set experiment
            env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
            if env_exp_id is not None:
                self._experiment_id = mlflow.set_experiment(experiment_id=env_exp_id).experiment_id
            else:
                self._experiment_id = mlflow.set_experiment(experiment_name=self.experiment_name).experiment_id

            # start run
            env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
            if env_run_id is not None:
                self._run_id = env_run_id
            else:
                self._run_id = self._mlflow_client.create_run(experiment_id=self._experiment_id)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        import mlflow
        import time
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

            # import time
            # before = time.time()
            # Convert all metrics to floats to placate mlflow.
            # mlflow.log_metrics(metrics=metrics, step=step)
            # after = time.time()
            # size = len(metrics)
            # latency = (after - before)
            # mlflow.log_metrics({
            #     "mlflow_batch_latency_s": latency,
            #     "mlflow_batch_size": size
            # })

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        import mlflow
        if self._enabled:
            self._mlflow_client.log_params(
                run_id=self._run_id,
                params=hyperparameters,
            )
            self._mlflow_client.flush(synchronous=False)

    def post_close(self):
        import mlflow
        if self._enabled:
            self._mlflow_client.set_terminated(self._run_id)
            self._mlflow_client.flush(synchronous=True)
