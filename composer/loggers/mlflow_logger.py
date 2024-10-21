# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `MLflow <https://www.mlflow.org/docs/latest/index.html>."""

from __future__ import annotations

import fnmatch
import logging
import multiprocessing
import os
import pathlib
import posixpath
import signal
import sys
import textwrap
import time
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import numpy as np
import torch

from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from mlflow import ModelVersion  # pyright: ignore[reportGeneralTypeIssues]

log = logging.getLogger(__name__)

__all__ = ['MLFlowLogger']

DEFAULT_MLFLOW_EXPERIMENT_NAME = 'my-mlflow-experiment'


class MlflowMonitorProcess(multiprocessing.Process):

    def __init__(self, main_pid, mlflow_run_id, mlflow_tracking_uri):
        super().__init__()
        self.main_pid = main_pid
        self.mlflow_run_id = mlflow_run_id
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.exit_event = multiprocessing.Event()
        self.crash_event = multiprocessing.Event()

    def handle_sigterm(self, signum, frame):
        from mlflow import MlflowClient
        client = MlflowClient(self.mlflow_tracking_uri)
        if client.get_run(self.mlflow_run_id).info.status == 'RUNNING':
            # Set the run status as KILLED if SIGTERM is received while the MLflow run is still
            # in status RUNNING.
            client.set_terminated(self.mlflow_run_id, status='KILLED')

    def run(self):
        from mlflow import MlflowClient

        os.setsid()
        # Register the signal handler in the child process
        signal.signal(signal.SIGTERM, self.handle_sigterm)

        while not self.exit_event.wait(10):
            try:
                # Signal 0 does not kill the process but performs error checking
                os.kill(self.main_pid, 0)
            except OSError:
                client = MlflowClient(self.mlflow_tracking_uri)
                client.set_terminated(self.mlflow_run_id, status='FAILED')
                break

        if self.crash_event.is_set():
            client = MlflowClient(self.mlflow_tracking_uri)
            client.set_terminated(self.mlflow_run_id, status='FAILED')

    def stop(self):
        self.exit_event.set()

    def crash(self):
        self.crash_event.set()
        self.exit_event.set()


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
        rename_metrics (dict[str, str], optional): A dict to rename metrics, requires an exact match on the key (default: ``None``)
        ignore_metrics (list[str], optional): A list of glob patterns for metrics to ignore when logging. (default: ``None``)
        ignore_hyperparameters (list[str], optional): A list of glob patterns for hyperparameters to ignore when logging. (default: ``None``)
        run_group (str, optional): A string to group runs together. (default: ``None``)
        resume (bool, optional): If ``True``, Composer will search for an existing run tagged with
            the `run_name` and resume it. If no existing run is found, a new run will be created.
            If ``False``, Composer will create a new run. (default: ``False``)
        logging_buffer_seconds (int, optional): The amount of time, in seconds, that MLflow
            waits before sending logs to the MLflow tracking server. Metrics/params/tags logged
            within this buffer time will be grouped in batches before being sent to the backend.
    """

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, Any]] = None,
        tracking_uri: Optional[Union[str, pathlib.Path]] = None,
        rank_zero_only: bool = True,
        flush_interval: int = 10,
        model_registry_prefix: str = '',
        model_registry_uri: Optional[str] = None,
        synchronous: bool = False,
        log_system_metrics: bool = True,
        rename_metrics: Optional[dict[str, str]] = None,
        ignore_metrics: Optional[list[str]] = None,
        ignore_hyperparameters: Optional[list[str]] = None,
        run_group: Optional[str] = None,
        resume: bool = False,
        logging_buffer_seconds: Optional[int] = 10,
    ) -> None:
        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='mlflow',
                conda_package='mlflow',
                conda_channel='conda-forge',
            ) from e
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_group = run_group
        self.tags = tags or {}
        self.model_registry_prefix = model_registry_prefix
        self.model_registry_uri = model_registry_uri
        self.synchronous = synchronous
        self.log_system_metrics = log_system_metrics
        self.rename_metrics = {} if rename_metrics is None else rename_metrics
        self.ignore_metrics = [] if ignore_metrics is None else ignore_metrics
        self.ignore_hyperparameters = [] if ignore_hyperparameters is None else ignore_hyperparameters
        if self.model_registry_uri == 'databricks-uc':
            if len(self.model_registry_prefix.split('.')) != 2:
                raise ValueError(
                    f'When registering to Unity Catalog, model_registry_prefix must be in the format ' +
                    f'{{catalog_name}}.{{schema_name}}, but got {self.model_registry_prefix}',
                )
        self.resume = resume

        if logging_buffer_seconds:
            os.environ['MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS'] = str(logging_buffer_seconds,)

        if log_system_metrics:
            # Set system metrics sampling interval and samples before logging so that system metrics
            # are collected every 5s, and aggregated over 6 samples before being logged
            # (logging per 30s).
            mlflow.set_system_metrics_samples_before_logging(6)
            mlflow.set_system_metrics_sampling_interval(5)

        self._rank_zero_only = rank_zero_only
        self._last_flush_time = time.time()
        self._flush_interval = flush_interval

        self._experiment_id: Optional[str] = None
        self._run_id = None
        self.run_url = None

        if self._enabled:
            if tracking_uri is None and os.getenv('DATABRICKS_TOKEN') is not None:
                tracking_uri = 'databricks'
            if tracking_uri is None:
                tracking_uri = mlflow.get_tracking_uri()
            self.tracking_uri = str(tracking_uri)
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
            assert self.experiment_name is not None  # type hint

            if os.getenv(
                'DATABRICKS_TOKEN',
            ) is not None and not self.experiment_name.startswith((
                '/Users/',
                '/Shared/',
            )):
                try:
                    from databricks.sdk import WorkspaceClient
                except ImportError as e:
                    raise MissingConditionalImportError(
                        extra_deps_group='mlflow',
                        conda_package='databricks-sdk',
                        conda_channel='conda-forge',
                    ) from e
                databricks_username = WorkspaceClient().current_user.me().user_name or ''
                self.experiment_name = os.path.join(
                    '/Users',
                    databricks_username,
                    self.experiment_name.strip('/'),
                )

            self._mlflow_client = MlflowClient(self.tracking_uri)
            # Set experiment
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
                    self._experiment_id = self._mlflow_client.create_experiment(name=self.experiment_name)

    def _start_mlflow_run(self, state):
        import mlflow

        # This function is only called if self._enabled is True, and therefore self._experiment_id is not None.
        assert self._experiment_id is not None

        env_run_id = os.getenv(
            mlflow.environment_variables.MLFLOW_RUN_ID.name,  # pyright: ignore[reportGeneralTypeIssues]
            None,
        )
        if env_run_id is not None:
            self._run_id = env_run_id
        elif self.resume:
            # Search for an existing run tagged with this Composer run if `self.resume=True`.
            run_name = self.tags['run_name']
            existing_runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=f'tags.run_name = "{run_name}"',
                output_format='list',
            )

            if len(existing_runs) > 0:
                self._run_id = existing_runs[0].info.run_id
                log.debug(f'Resuming mlflow run with run id: {self._run_id}')
            else:
                log.debug(
                    'Creating a new mlflow run as `resume` was set to True but no previous run was '
                    'found.',
                )
                new_run = self._mlflow_client.create_run(
                    experiment_id=self._experiment_id,
                    run_name=self.run_name,
                )
                self._run_id = new_run.info.run_id
        else:
            # Create a new run if `env_run_id` is not set or `self.resume=False`.
            new_run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id,
                run_name=self.run_name,
            )
            self._run_id = new_run.info.run_id

        tags = self.tags or {}
        if self.run_group:
            tags['run_group'] = self.run_group
        mlflow.start_run(
            run_id=self._run_id,
            tags=self.tags,
            log_system_metrics=self.log_system_metrics,
        )
        if self.tracking_uri == 'databricks':
            # Start a background process to monitor the job to report the job status to MLflow.
            self.monitor_process = MlflowMonitorProcess(
                os.getpid(),
                self._run_id,
                self.tracking_uri,
            )
            self.monitor_process.start()

    def _global_exception_handler(self, original_excepthook, exc_type, exc_value, exc_traceback):
        """Catch global exception."""
        self._global_exception_occurred += 1
        original_excepthook(exc_type, exc_value, exc_traceback)

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused

        if self.run_name is None:
            self.run_name = state.run_name

        self._global_exception_occurred = 0

        # Store the Composer run name in the MLFlow run tags so it can be retrieved for autoresume
        self.tags['run_name'] = os.environ.get('RUN_NAME', state.run_name)

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.run_name += f'-rank{dist.get_global_rank()}'

        # Register the global exception handler so that uncaught exception is tracked.
        original_excepthook = sys.excepthook
        sys.excepthook = lambda exc_type, exc_value, exc_traceback: self._global_exception_handler(
            original_excepthook,
            exc_type,
            exc_value,
            exc_traceback,
        )
        # Start run
        if self._enabled:
            self._start_mlflow_run(state)

        # If rank zero only, broadcast the MLFlow experiment and run IDs to other ranks, so the MLFlow run info is
        # available to other ranks during runtime.
        if self._rank_zero_only:
            mlflow_ids_list = [self._experiment_id, self._run_id]
            dist.broadcast_object_list(mlflow_ids_list, src=0)
            self._experiment_id, self._run_id = mlflow_ids_list

    def after_load(self, state: State, logger: Logger) -> None:
        logger.log_hyperparameters({
            'mlflow_experiment_id': self._experiment_id,
            'mlflow_run_id': self._run_id,
        })
        self.run_url = posixpath.join(
            os.environ.get('DATABRICKS_HOST', ''),
            'ml',
            'experiments',
            str(self._experiment_id),
            'runs',
            str(self._run_id),
        )

    def log_table(
        self,
        columns: list[str],
        rows: list[list[Any]],
        name: str = 'Table',
        step: Optional[int] = None,
    ) -> None:
        del step
        if self._enabled:
            try:
                import pandas as pd
            except ImportError as e:
                raise MissingConditionalImportError(
                    extra_deps_group='pandas',
                    conda_package='pandas',
                    conda_channel='conda-forge',
                ) from e
            table = pd.DataFrame.from_records(data=rows, columns=columns)
            assert isinstance(self._run_id, str)
            self._mlflow_client.log_table(
                run_id=self._run_id,
                data=table,
                artifact_file=f'{name}.json',
            )

    def rename(self, key: str):
        return self.rename_metrics.get(key, key)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        from mlflow import log_metrics

        if self._enabled:
            # Convert all metrics to floats to placate mlflow.
            metrics = {
                self.rename(k): float(v)
                for k, v in metrics.items()
                if not any(fnmatch.fnmatch(k, pattern) for pattern in self.ignore_metrics)
            }
            log_metrics(
                metrics=metrics,
                step=step,
                synchronous=self.synchronous,
            )

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        from mlflow import log_params

        if self._enabled:
            hyperparameters = {
                k: v
                for k, v in hyperparameters.items()
                if not any(fnmatch.fnmatch(k, pattern) for pattern in self.ignore_hyperparameters)
            }
            log_params(
                params=hyperparameters,
                synchronous=self.synchronous,
            )

    def register_model(
        self,
        model_uri: str,
        name: str,
        await_registration_for: int = 300,
        tags: Optional[dict[str, Any]] = None,
    ) -> 'ModelVersion':
        """Register a model to model registry.

        Args:
            model_uri (str): The URI of the model to register.
            name (str): The name of the model to register. Will be appended to ``model_registry_prefix``.
            await_registration_for (int, optional): The number of seconds to wait for the model to be registered.
                Defaults to 300.
            tags (Optional[dict[str, Any]], optional): A dictionary of tags to add to the model. Defaults to None.
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

    def save_model(self, flavor: Literal['transformers', 'peft'], **kwargs):
        """Save a model to MLflow.

        Note: The ``'peft'`` flavor is experimental and the API is subject to change without warning.

        Args:
            flavor (Literal['transformers', 'peft']): The MLflow model flavor to use. Currently only ``'transformers'`` and ``'peft'`` are supported.
            **kwargs: Keyword arguments to pass to the MLflow model saving function.

        Raises:
            NotImplementedError: If ``flavor`` is not ``'transformers'`` or ``'peft'``.
        """
        if self._enabled:
            import mlflow

            if flavor == 'transformers':
                mlflow.transformers.save_model(**kwargs)
            elif flavor == 'peft':
                import transformers

                # TODO: Remove after mlflow fixes the bug that makes this necessary
                mlflow.store._unity_catalog.registry.rest_store.get_feature_dependencies = lambda *args, **kwargs: ''  # type: ignore

                # This is a temporary workaround until MLflow adds full support for saving PEFT models.
                # https://github.com/mlflow/mlflow/issues/9256
                log.warning(
                    'Saving PEFT models using MLflow is experimental and the API is subject to change without warning.',
                )
                expected_keys = {'path', 'save_pretrained_dir'}
                if not expected_keys.issubset(kwargs.keys()):
                    raise ValueError(f'Expected keys {expected_keys} but got {kwargs.keys()}')

                # This does not implement predict for now, as we will wait for the full MLflow support
                # for PEFT models.
                class PeftModel(mlflow.pyfunc.PythonModel):

                    def load_context(self, context):
                        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                            context.artifacts['lora_checkpoint'],
                        )
                        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                            context.artifacts['lora_checkpoint'],
                        )

                from mlflow.models.signature import ModelSignature
                from mlflow.types import ColSpec, DataType, Schema

                # This is faked for now, until MLflow adds full support for saving PEFT models.
                input_schema = Schema([
                    ColSpec(DataType.string, 'fake_input'),
                ])
                output_schema = Schema([ColSpec(DataType.string)])
                signature = ModelSignature(inputs=input_schema, outputs=output_schema)

                # Symlink the directory so that we control the path that MLflow saves the model under
                os.symlink(kwargs['save_pretrained_dir'], 'lora_checkpoint')

                mlflow.pyfunc.save_model(
                    path=kwargs['path'],
                    artifacts={'lora_checkpoint': 'lora_checkpoint'},
                    python_model=PeftModel(),
                    signature=signature,
                )

                os.unlink('lora_checkpoint')
            else:
                raise NotImplementedError(f'flavor {flavor} not supported.')

    def log_model(self, flavor: Literal['transformers'], **kwargs):
        """Log a model to MLflow.

        Args:
            flavor (Literal['transformers']): The MLflow model flavor to use. Currently only ``'transformers'`` is supported.
            **kwargs: Keyword arguments to pass to the MLflow model logging function.

        Raises:
            NotImplementedError: If ``flavor`` is not ``'transformers'``.
        """
        if self._enabled:
            import mlflow

            if flavor == 'transformers':
                mlflow.transformers.log_model(**kwargs)
            else:
                raise NotImplementedError(f'flavor {flavor} not supported.')

    def register_model_with_run_id(
        self,
        model_uri: str,
        name: str,
        await_creation_for: int = 300,
        tags: Optional[dict[str, Any]] = None,
    ):
        """Similar to ``register_model``, but uses a different MLflow API to allow passing in the run id.

        Args:
            model_uri (str): The URI of the model to register.
            name (str): The name of the model to register. Will be appended to ``model_registry_prefix``.
            await_creation_for (int, optional): The number of seconds to wait for the model to be registered. Defaults to 300.
            tags (Optional[dict[str, Any]], optional): A dictionary of tags to add to the model. Defaults to None.
        """
        if self._enabled:
            from mlflow.exceptions import MlflowException
            from mlflow.protos.databricks_pb2 import (
                ALREADY_EXISTS,
                RESOURCE_ALREADY_EXISTS,
                ErrorCode,
            )

            full_name = f'{self.model_registry_prefix}.{name}' if len(self.model_registry_prefix) > 0 else name

            # This try/catch code is copied from
            # https://github.com/mlflow/mlflow/blob/3ba1e50e90a38be19920cb9118593a43d7cfa90e/mlflow/tracking/_model_registry/fluent.py#L90-L103
            try:
                create_model_response = self._mlflow_client.create_registered_model(full_name)
                log.info(f'Successfully registered model {name} with {create_model_response.name}')
            except MlflowException as e:
                if e.error_code in (
                    ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
                    ErrorCode.Name(ALREADY_EXISTS),
                ):
                    log.info(f'Registered model {name} already exists. Creating a new version of this model...')
                else:
                    raise e

            create_version_response = self._mlflow_client.create_model_version(
                name=full_name,
                source=model_uri,
                run_id=self._run_id,
                await_creation_for=await_creation_for,
                tags=tags,
            )

            log.info(
                f'Successfully created model version {create_version_response.version} for model {create_version_response.name}',
            )

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'image',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[dict[int, str]] = None,
        use_table: bool = True,
    ):
        unused_args = (masks, mask_class_labels)  # Unused (only for wandb)
        if any(unused_args):
            warnings.warn(
                textwrap.dedent(
                    f"""MLFlowLogger does not support masks, class labels, or tables of images,
                          but got masks={masks}, mask_class_labels={mask_class_labels}""",
                ),
            )
        if self._enabled:
            if not isinstance(images, Sequence) and images.ndim <= 3:
                images = [images]
            for im_ind, image in enumerate(images):
                image = _convert_to_mlflow_image(image, channels_last)
                assert isinstance(self._run_id, str)
                self._mlflow_client.log_image(
                    image=image,
                    key=f'{name}_{im_ind}',
                    run_id=self._run_id,
                    step=step,
                )

    def post_close(self):
        if self._enabled:
            if hasattr(self, 'monitor_process'):
                # Check if there is an uncaught exception, which means `post_close()` is triggered
                # due to program crash.
                finish_with_exception = self._global_exception_occurred == 1
                if finish_with_exception:
                    self.monitor_process.crash()
                    return

                # Stop the monitor process since it's entering the cleanup phase.
                self.monitor_process.stop()

            import mlflow

            assert isinstance(self._run_id, str)

            mlflow.flush_async_logging()
            exc_tpe, exc_info, tb = sys.exc_info()
            if (exc_tpe, exc_info, tb) == (None, None, None):
                current_status = self._mlflow_client.get_run(self._run_id).info.status
                if current_status == 'RUNNING':
                    self._mlflow_client.set_terminated(self._run_id, status='FINISHED')
            else:
                # Record there was an error
                self._mlflow_client.set_terminated(self._run_id, status='FAILED')

            mlflow.end_run()
            if hasattr(self, 'monitor_process'):
                self.monitor_process.join()


def _convert_to_mlflow_image(
    image: Union[np.ndarray, torch.Tensor],
    channels_last: bool,
) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()

    # Error out for empty arrays or weird arrays of dimension 0.
    if np.any(np.equal(image.shape, 0)):
        raise ValueError(f'Got an image (shape {image.shape}) with at least one dimension being 0!')

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
            textwrap.dedent(
                f'''Input image must be 3 dimensions, but instead
                            got {image.ndim} dims at shape: {image.shape}
                            Your input image was interpreted as a batch of {image.ndim}
                            -dimensional images because you either specified a
                            {image.ndim + 1}D image or a list of {image.ndim}D images.
                            Please specify either a 4D image of a list of 3D images''',
            ),
        )

    assert isinstance(image, np.ndarray)
    if not channels_last:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] not in [1, 3, 4]:
        raise ValueError(
            textwrap.dedent(
                f'''Input image must have 1, 3, or 4 channels, but instead
                            got {image.shape[-1]} channels at shape: {image.shape}
                            Please specify either a 1-, 3-, or 4-channel image or a list of
                            1-, 3-, or 4-channel images''',
            ),
        )
    return image
