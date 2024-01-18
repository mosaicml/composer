# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `MLflow <https://www.mlflow.org/docs/latest/index.html>."""

from __future__ import annotations

import fnmatch
import os
import pathlib
import textwrap
import time
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from mlflow import ModelVersion  # pyright: ignore[reportGeneralTypeIssues]

__all__ = ['MLFlowLogger']

DEFAULT_MLFLOW_EXPERIMENT_NAME = 'my-mlflow-experiment'


class MLFlowLogger(LoggerDestination):
    """Log to `MLflow <https://www.mlflow.org/docs/latest/index.html>`_.

    Args:
        experiment_name: (str, optional): MLflow experiment name. If not set it will be
            use the MLflow environment variable or a default value
        run_name: (str, optional): MLflow run name. If not set it will be the same as the
            Trainer run name
        tags: (dict, optional): MLflow tags to log with the run
        tracking_uri (str | pathlib.Path, optional): MLflow tracking uri, the URI to the
            remote or local endpoint where logs are stored (If none it is set to MLflow default)
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process
            (default: ``True``).
        flush_interval (int): The amount of time, in seconds, that MLflow must wait between
            logging batches of metrics. Any metrics that are recorded by Composer during
            this interval are enqueued, and the queue is flushed when the interval elapses
            (default: ``10``).
        model_registry_prefix (str, optional): The prefix to use when registering models.
            If registering to Unity Catalog, must be in the format ``{catalog_name}.{schema_name}``.
            (default: `''`)
        model_registry_uri (str, optional): The URI of the model registry to use. To register models
            to Unity Catalog, set to ``databricks-uc``. (default: None)
        synchronous (bool, optional): Whether to log synchronously. If ``True``, Mlflow will log
            synchronously to the MLflow backend. If ``False``, Mlflow will log asynchronously. (default: ``False``)
        log_system_metrics (bool, optional): Whether to log system metrics. If ``True``, Mlflow will
            log system metrics (CPU/GPU/memory/network usage) during training. (default: ``True``)
        ignore_metrics (List[str], optional): A list of glob patterns for metrics to ignore when logging. (default: ``None``)
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        tracking_uri: Optional[Union[str, pathlib.Path]] = None,
        rank_zero_only: bool = True,
        flush_interval: int = 10,
        model_registry_prefix: str = '',
        model_registry_uri: Optional[str] = None,
        synchronous: bool = False,
        log_system_metrics: bool = True,
        ignore_metrics: Optional[List[str]] = None,
    ) -> None:
        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='mlflow',
                                                conda_package='mlflow',
                                                conda_channel='conda-forge') from e
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags
        self.model_registry_prefix = model_registry_prefix
        self.model_registry_uri = model_registry_uri
        self.synchronous = synchronous
        self.log_system_metrics = log_system_metrics
        self.ignore_metrics = [] if ignore_metrics is None else ignore_metrics
        if self.model_registry_uri == 'databricks-uc':
            if len(self.model_registry_prefix.split('.')) != 2:
                raise ValueError(f'When registering to Unity Catalog, model_registry_prefix must be in the format ' +
                                 f'{{catalog_name}}.{{schema_name}}, but got {self.model_registry_prefix}')

        self._rank_zero_only = rank_zero_only
        self._last_flush_time = time.time()
        self._flush_interval = flush_interval

        self._experiment_id: Optional[str] = None
        self._run_id = None

        if self._enabled:
            self.tracking_uri = str(tracking_uri or mlflow.get_tracking_uri())
            mlflow.set_tracking_uri(self.tracking_uri)

            if self.model_registry_uri is not None:
                mlflow.set_registry_uri(self.model_registry_uri)
            # Set up MLflow state
            self._run_id = None
            if self.experiment_name is None:
                self.experiment_name = os.getenv(
                    mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.name,  # type: ignore
                    DEFAULT_MLFLOW_EXPERIMENT_NAME,
                )
            self._mlflow_client = MlflowClient(self.tracking_uri)
            # Set experiment.
            env_exp_id = os.getenv(
                mlflow.environment_variables.MLFLOW_EXPERIMENT_ID.name,  # pyright: ignore[reportGeneralTypeIssues]
                None,
            )
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

        # Store the Composer run name in the MLFlow run tags so it can be retrieved for autoresume.
        self.tags = self.tags or {}
        self.tags['composer_run_name'] = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.run_name += f'-rank{dist.get_global_rank()}'

        # Start run
        if self._enabled:
            env_run_id = os.getenv(
                mlflow.environment_variables.MLFLOW_RUN_ID.name,  # pyright: ignore[reportGeneralTypeIssues]
                None,
            )
            if env_run_id is not None:
                self._run_id = env_run_id
            else:
                # Search for an existing run tagged with this Composer run.
                assert self._experiment_id is not None
                existing_runs = mlflow.search_runs(experiment_ids=[self._experiment_id],
                                                   filter_string=f'tags.composer_run_name = "{state.run_name}"',
                                                   output_format='list')
                if len(existing_runs) > 0:
                    self._run_id = existing_runs[0].info.run_id
                else:
                    new_run = self._mlflow_client.create_run(
                        experiment_id=self._experiment_id,
                        run_name=self.run_name,
                    )
                    self._run_id = new_run.info.run_id
            mlflow.start_run(
                run_id=self._run_id,
                tags=self.tags,
                log_system_metrics=self.log_system_metrics,
            )

        # If rank zero only, broadcast the MLFlow experiment and run IDs to other ranks, so the MLFlow run info is
        # available to other ranks during runtime.
        if self._rank_zero_only:
            mlflow_ids_list = [self._experiment_id, self._run_id]
            dist.broadcast_object_list(mlflow_ids_list, src=0)
            self._experiment_id, self._run_id = mlflow_ids_list

    def after_load(self, state: State, logger: Logger) -> None:
        logger.log_hyperparameters({'mlflow_experiment_id': self._experiment_id, 'mlflow_run_id': self._run_id})

    def log_table(self, columns: List[str], rows: List[List[Any]], name: str = 'Table') -> None:
        if self._enabled:
            try:
                import pandas as pd
            except ImportError as e:
                raise MissingConditionalImportError(extra_deps_group='pandas',
                                                    conda_package='pandas',
                                                    conda_channel='conda-forge') from e
            table = pd.DataFrame.from_records(data=rows, columns=columns)
            assert isinstance(self._run_id, str)
            self._mlflow_client.log_table(
                run_id=self._run_id,
                data=table,
                artifact_file=f'{name}.json',
            )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        from mlflow import log_metrics
        if self._enabled:
            # Convert all metrics to floats to placate mlflow.
            metrics = {
                k: float(v)
                for k, v in metrics.items()
                if not any(fnmatch.fnmatch(k, pattern) for pattern in self.ignore_metrics)
            }
            log_metrics(
                metrics=metrics,
                step=step,
                synchronous=self.synchronous,
            )

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        from mlflow import log_params

        if self._enabled:
            log_params(
                params=hyperparameters,
                synchronous=self.synchronous,
            )

    def register_model(
        self,
        model_uri: str,
        name: str,
        await_registration_for: int = 300,
        tags: Optional[Dict[str, Any]] = None,
    ) -> 'ModelVersion':
        """Register a model to model registry.

        Args:
            model_uri (str): The URI of the model to register.
            name (str): The name of the model to register. Will be appended to ``model_registry_prefix``.
            await_registration_for (int, optional): The number of seconds to wait for the model to be registered.
                Defaults to 300.
            tags (Optional[Dict[str, Any]], optional): A dictionary of tags to add to the model. Defaults to None.
            registry_uri (str, optional): The URI of the model registry. Defaults to `None` which will register to
                the Databricks Unity Catalog.

        Returns:
            ModelVersion: The registered model.
        """
        if self._enabled:
            full_name = f'{self.model_registry_prefix}.{name}' if len(self.model_registry_prefix) > 0 else name

            import mlflow
            return mlflow.register_model(
                model_uri=model_uri,
                name=full_name,
                await_registration_for=await_registration_for,
                tags=tags,
            )

    def save_model(self, flavor: str, **kwargs):
        """Save a model to MLflow.

        Args:
            flavor (str): The MLflow model flavor to use. Currently only ``'transformers'`` is supported.
            **kwargs: Keyword arguments to pass to the MLflow model saving function.

        Raises:
            NotImplementedError: If ``flavor`` is not ``'transformers'``.
        """
        if self._enabled:
            import mlflow
            if flavor == 'transformers':
                mlflow.transformers.save_model(**kwargs,)
            else:
                raise NotImplementedError(f'flavor {flavor} not supported.')

    def log_model(self, flavor: str, **kwargs):
        """Log a model to MLflow.

        Args:
            flavor (str): The MLflow model flavor to use. Currently only ``'transformers'`` is supported.
            **kwargs: Keyword arguments to pass to the MLflow model logging function.

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
                assert isinstance(self._run_id, str)
                self._mlflow_client.log_image(image=image,
                                              artifact_file=f'{name}_{step}_{im_ind}.png',
                                              run_id=self._run_id)

    def post_close(self):
        if self._enabled:
            import mlflow

            assert isinstance(self._run_id, str)
            self._mlflow_client.set_terminated(self._run_id)
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
    if image.shape[-1] not in [1, 3, 4]:
        raise ValueError(
            textwrap.dedent(f'''Input image must have 1, 3, or 4 channels, but instead
                            got {image.shape[-1]} channels at shape: {image.shape}
                            Please specify either a 1-, 3-, or 4-channel image or a list of
                            1-, 3-, or 4-channel images'''))
    return image
