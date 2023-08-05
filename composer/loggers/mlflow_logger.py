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
        flush_interval (int): The amount of time, in seconds, that MLflow must wait between
            logging batches of metrics. Any metrics that are recorded by Composer during
            this interval are enqueued, and the queue is flushed when the interval elapses
            (default: ``10``).
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tracking_uri: Optional[Union[str, pathlib.Path]] = None,
        rank_zero_only: bool = True,
        flush_interval: int = 10,
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
        self._last_flush_time = time.time()
        self._flush_interval = flush_interval
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
            self._mlflow_client = MlflowClient(self.tracking_uri)
            # Create an instance of MlflowAutologgingQueueingClient - an optimized version
            # of MlflowClient - that automatically batches metrics together and supports
            # asynchronous logging for improved performance
            self._optimized_mlflow_client = MlflowAutologgingQueueingClient(self.tracking_uri)

            # set experiment. we use MlflowClient for experiment retrieval and creation
            # because MlflowAutologgingQueueingClient doesn't support it
            env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
            if env_exp_id is not None:
                self._experiment_id = env_exp_id
            elif exp := self._mlflow_client.get_experiment_by_name(name=self.experiment_name):
                self._experiment_id = exp.experiment_id
            else:
                self._experiment_id = (
                    self._mlflow_client.create_experiment(name=self.experiment_name)
                )

            # start run
            env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
            if env_run_id is not None:
                self._run_id = env_run_id
            else:
                new_run = self._mlflow_client.create_run(
                    experiment_id=self._experiment_id,
                    run_name=self.run_name,
                )
                self._run_id = new_run.info.run_id

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            # Convert all metrics to floats to placate mlflow.
            metrics = {k: float(v) for k, v in metrics.items()}
            self._optimized_mlflow_client.log_metrics(
                run_id=self._run_id,
                metrics=metrics,
                step=step,
            )
            time_since_flush = time.time() - self._last_flush_time
            if time_since_flush >= self._flush_interval:
                self._optimized_mlflow_client.flush(synchronous=False)
                self._last_flush_time = time.time()

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            self._optimized_mlflow_client.log_params(
                run_id=self._run_id,
                params=hyperparameters,
            )
            self._optimized_mlflow_client.flush(synchronous=False)

    def post_close(self):
        if self._enabled:
            # we use MlflowClient for run termination because MlflowAutologgingQueueingClient's
            # run termination relies on scheduling Python futures, which is not supported within
            # the Python atexit handler in which post_close() is called
            self._mlflow_client.set_terminated(self._run_id)

    def _flush(self):
        """Test-only method to synchronously flush all queued metrics"""
        return self._optimized_mlflow_client.flush(synchronous=True)
