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
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.entities import Metric, Param
from mlflow.utils.time_utils import get_current_time_millis

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
        self.tracking_uri = tracking_uri
        self._mlflow_client = mlflow.MlflowClient(tracking_uri)

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

            # start run
            if dist.get_global_rank() == 0:
                if self.tracking_uri is not None:
                    mlflow.set_tracking_uri(self.tracking_uri)
                # set experiment
                env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
                if env_exp_id is not None:
                    mlflow.set_experiment(experiment_id=env_exp_id)
                else:
                    parent_exp_id = mlflow.set_experiment(experiment_name=self.experiment_name).experiment_id
                env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
                if env_run_id is None:
                    parent_run = mlflow.start_run(run_name=self.run_name)
                    parent_run_id = parent_run.info.run_id
                else:
                    mlflow.start_run(run_id=env_run_id, run_name=self.run_name)
                    parent_run_id = env_run_id
                mlflow.end_run()

            else:
                parent_run_id = None
                parent_exp_id = None

            run_id_list = [parent_run_id, parent_exp_id]
            dist.broadcast_object_list(run_id_list, src=0)
            parent_run_id = run_id_list[0]
            parent_exp_id = run_id_list[1]
            # mlflow.set_experiment(experiment_id=parent_exp_id)
            # mlflow.set_tag(MLFLOW_PARENT_RUN_ID, parent_run_id)
            self.run = self._mlflow_client.create_run(
                experiment_id=parent_exp_id,
                run_name=self.run_name,
                tags={MLFLOW_PARENT_RUN_ID: parent_run_id})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            # Convert all metrics to floats to placate mlflow.
            metrics = {k: float(v) for k, v in metrics.items()}
            timestamp = get_current_time_millis()
            metrics_arr = [Metric(key, value, timestamp, step or 0) for key, value in metrics.items()]
            self._mlflow_client.log_batch(
                run_id=self.run.info.run_id,
                metrics=metrics_arr,
                params=[],
                tags=[])

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            params_arr = [Param(key, str(value)) for key, value in hyperparameters.items()]
            self._mlflow_client.log_batch(
                run_id=self.run.info.run_id, metrics=[], params=params_arr, tags=[])

    def post_close(self):
        import mlflow
        if self._enabled:
            mlflow.end_run()
            if dist.get_global_rank() == 0:
                mlflow.end_run()
