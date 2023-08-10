# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `MLFlow <https://www.mlflow.org/docs/latest/index.html>."""

from __future__ import annotations

import os
import pathlib
import textwrap
from typing import Any, Dict, Optional, Sequence, Union

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
            if self.tracking_uri is not None:
                mlflow.set_tracking_uri(self.tracking_uri)

            # set experiment
            env_exp_id = os.getenv(mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name, None)
            if env_exp_id is not None:
                mlflow.set_experiment(experiment_id=env_exp_id)
            else:
                mlflow.set_experiment(experiment_name=self.experiment_name)

            # start run
            env_run_id = os.getenv(mlflow.environment_variables.MLFLOW_RUN_ID.name, None)
            if env_run_id is not None:
                mlflow.start_run(run_id=env_run_id)
            else:
                mlflow.start_run(run_name=self.run_name)

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
        del masks, mask_class_labels, use_table  # Unused (only for wandb)
        import mlflow
        if self._enabled:
            if not isinstance(images, Sequence) and images.ndim <= 3:
                images = [images]
            for im_ind, image in enumerate(images):
                image = _convert_to_mlflow_image(image, channels_last)
                mlflow.log_image(image, artifact_file=f'{name}_{step}_{im_ind}.png')

    def post_close(self):
        import mlflow
        if self._enabled:
            mlflow.end_run()


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
    return image
