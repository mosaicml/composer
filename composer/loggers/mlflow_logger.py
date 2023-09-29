# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `MLFlow <https://www.mlflow.org/docs/latest/index.html>."""

from __future__ import annotations

import os
import pathlib
import textwrap
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

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
            from mlflow import MlflowClient
            from mlflow.utils.autologging_utils import MlflowAutologgingQueueingClient
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='mlflow',
                                                conda_package='mlflow',
                                                conda_channel='conda-forge') from e
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self.run_name = run_name
        self.experiment_name = experiment_name
        self._rank_zero_only = rank_zero_only
        self._last_flush_time = time.time()
        self._flush_interval = flush_interval
        if self._enabled:
            self.tracking_uri = str(tracking_uri or mlflow.get_tracking_uri())
            mlflow.set_tracking_uri(self.tracking_uri)
            # Set up MLflow state
            self._run_id = None
            if self.experiment_name is None:
                self.experiment_name = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.name,
                                                 DEFAULT_MLFLOW_EXPERIMENT_NAME)
            self._mlflow_client = MlflowClient(self.tracking_uri)
            # Create an instance of MlflowAutologgingQueueingClient - an optimized version
            # of MlflowClient - that automatically batches metrics together and supports
            # asynchronous logging for improved performance
            self._optimized_mlflow_client = MlflowAutologgingQueueingClient(self.tracking_uri)
            # Set experiment. We use MlflowClient for experiment retrieval and creation
            # because MlflowAutologgingQueueingClient doesn't support it
            env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
            if env_exp_id is not None:
                self._experiment_id = env_exp_id
            else:
                exp_from_name = self._mlflow_client.get_experiment_by_name(name=self.experiment_name)
                if exp_from_name is not None:
                    self._experiment_id = exp_from_name.experiment_id
                else:
                    self._experiment_id = (self._mlflow_client.create_experiment(name=self.experiment_name))

    def init(self, state: State, logger: Logger) -> None:
        import mlflow
        del logger  # unused

        if self.run_name is None:
            self.run_name = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.run_name += f'-rank{dist.get_global_rank()}'

        # Start run
        if self._enabled:
            env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
            if env_run_id is not None:
                self._run_id = env_run_id
            else:
                new_run = self._mlflow_client.create_run(
                    experiment_id=self._experiment_id,
                    run_name=self.run_name,
                )
                self._run_id = new_run.info.run_id
            mlflow.start_run(run_id=self._run_id)

    def log_table(self, columns: List[str], rows: List[List[Any]], name: str = 'Table') -> None:
        if self._enabled:
            try:
                import pandas as pd
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='pandas',
                                                    conda_package='pandas',
                                                    conda_channel='conda-forge') from e
            table = pd.DataFrame.from_records(data=rows, columns=columns)
            self._mlflow_client.log_table(
                run_id=self._run_id,
                data=table,
                artifact_file=f'{name}.json',
            )

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

    def log_model(self, flavor: str, **kwargs):
        """Log a model to MLFlow.

        Args:
            flavor (str): The MLFlow model flavor to use. Currently only ``'transformers'`` is supported.
            **kwargs: Keyword arguments to pass to the MLFlow model logging function.

        Raises:
            NotImplementedError: If ``flavor`` is not ``'transformers'``.
        """
        if self._enabled:
            import mlflow
            if flavor == 'transformers':
                mlflow.transformers.log_model(**kwargs,)
            else:
                raise NotImplementedError(f'flavor {flavor} not supported.')

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'image',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[Dict[int, str]] = None,
        use_table: bool = True,
    ):
        unused_args = (masks, mask_class_labels)  # Unused (only for wandb)
        if any(unused_args):
            warnings.warn(
                textwrap.dedent(f"""MLFlowLogger does not support masks, class labels, or tables of images,
                          but got masks={masks}, mask_class_labels={mask_class_labels}"""))
        if self._enabled:
            if not isinstance(images, Sequence) and images.ndim <= 3:
                images = [images]
            for im_ind, image in enumerate(images):
                image = _convert_to_mlflow_image(image, channels_last)
                self._mlflow_client.log_image(image=image,
                                              artifact_file=f'{name}_{step}_{im_ind}.png',
                                              run_id=self._run_id)

    def post_close(self):
        if self._enabled:
            import mlflow

            # We use MlflowClient for run termination because MlflowAutologgingQueueingClient's
            # run termination relies on scheduling Python futures, which is not supported within
            # the Python atexit handler in which post_close() is called
            self._mlflow_client.set_terminated(self._run_id)
            mlflow.end_run()

    def _flush(self):
        """Test-only method to synchronously flush all queued metrics."""
        return self._optimized_mlflow_client.flush(synchronous=True)


def _convert_to_mlflow_image(image: Union[np.ndarray, torch.Tensor], channels_last: bool) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()

    # Error out for empty arrays or weird arrays of dimension 0.
    if np.any(np.equal(image.shape, 0)):
        raise ValueError(f'Got an image (shape {image.shape}) with at least one dimension being 0! ')

    # Squeeze any singleton dimensions and then add them back in if image dimension
    # less than 3.
    image = image.squeeze()

    # Add in length-one dimensions to get back up to 3
    # putting channels last.
    if image.ndim == 1:
        image = np.expand_dims(image, (1, 2))
        channels_last = True
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
        channels_last = True

    if image.ndim != 3:
        raise ValueError(
            textwrap.dedent(f'''Input image must be 3 dimensions, but instead
                            got {image.ndim} dims at shape: {image.shape}
                            Your input image was interpreted as a batch of {image.ndim}
                            -dimensional images because you either specified a
                            {image.ndim + 1}D image or a list of {image.ndim}D images.
                            Please specify either a 4D image of a list of 3D images'''))

    assert isinstance(image, np.ndarray)
    if not channels_last:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] not in [1, 3, 4]:
        raise ValueError(
            textwrap.dedent(f'''Input image must have 1, 3, or 4 channels, but instead
                            got {image.shape[-1]} channels at shape: {image.shape}
                            Please specify either a 1-, 3-, or 4-channel image or a list of
                            1-, 3-, or 4-channel images'''))
    return image
