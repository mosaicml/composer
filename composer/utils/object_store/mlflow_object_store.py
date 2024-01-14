# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""MLflow Artifacts object store."""

from __future__ import annotations

import logging
import os
import pathlib
import tempfile
from typing import Callable, List, Optional, Tuple, Union

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore, ObjectStoreTransientError

__all__ = ['MLFlowObjectStore']

MLFLOW_DATABRICKS_TRACKING_URI = 'databricks'
MLFLOW_DBFS_PATH_PREFIX = 'databricks/mlflow-tracking/'

DEFAULT_MLFLOW_EXPERIMENT_NAME = 'mlflow-object-store'

MLFLOW_EXPERIMENT_ID_FORMAT_KEY = 'mlflow_experiment_id'
MLFLOW_RUN_ID_FORMAT_KEY = 'mlflow_run_id'

MLFLOW_EXPERIMENT_ID_PLACEHOLDER = '{' + MLFLOW_EXPERIMENT_ID_FORMAT_KEY + '}'
MLFLOW_RUN_ID_PLACEHOLDER = '{' + MLFLOW_RUN_ID_FORMAT_KEY + '}'

log = logging.getLogger(__name__)


def _wrap_mlflow_exceptions(uri: str, e: Exception):
    """Wraps retryable MLflow errors in ObjectStoreTransientError for automatic retry handling."""
    from mlflow.exceptions import (ABORTED, DATA_LOSS, DEADLINE_EXCEEDED, ENDPOINT_NOT_FOUND, INTERNAL_ERROR,
                                   INVALID_STATE, NOT_FOUND, REQUEST_LIMIT_EXCEEDED, RESOURCE_DOES_NOT_EXIST,
                                   RESOURCE_EXHAUSTED, TEMPORARILY_UNAVAILABLE, ErrorCode, MlflowException)

    # https://github.com/mlflow/mlflow/blob/39b76b5b05407af5d223e892b03e450b7264576a/mlflow/exceptions.py for used error codes.
    # https://github.com/mlflow/mlflow/blob/39b76b5b05407af5d223e892b03e450b7264576a/mlflow/protos/databricks.proto for code descriptions.
    retryable_server_codes = [
        ErrorCode.Name(code) for code in [
            DATA_LOSS,
            INTERNAL_ERROR,
            INVALID_STATE,
            TEMPORARILY_UNAVAILABLE,
            DEADLINE_EXCEEDED,
        ]
    ]
    retryable_client_codes = [ErrorCode.Name(code) for code in [ABORTED, REQUEST_LIMIT_EXCEEDED, RESOURCE_EXHAUSTED]]
    not_found_codes = [ErrorCode.Name(code) for code in [RESOURCE_DOES_NOT_EXIST, NOT_FOUND, ENDPOINT_NOT_FOUND]]

    if isinstance(e, MlflowException):
        error_code = e.error_code  # pyright: ignore
        if error_code in retryable_server_codes or error_code in retryable_client_codes:
            raise ObjectStoreTransientError(error_code) from e
        elif error_code in not_found_codes:
            raise FileNotFoundError(f'Object {uri} not found') from e

    raise e


class MLFlowObjectStore(ObjectStore):
    """Utility class for uploading and downloading artifacts from MLflow.

    It can be initializd for an existing run, a new run in an existing experiment, the active run used by the `mlflow`
    module, or a new run in a new experiment. See the documentation for ``path`` for more details.

    .. note::
        At this time, only Databricks-managed MLflow with a 'databricks' tracking URI is supported.
        Using this object store requires configuring Databricks authentication through a configuration file or
        environment variables. See
        https://databricks-sdk-py.readthedocs.io/en/latest/authentication.html#databricks-native-authentication

        Unlike other object stores, the DBFS URI scheme for MLflow artifacts has no bucket, and the path is prefixed
        with the artifacts root directory for a given experiment/run,
        `databricks/mlflow-tracking/<experiment_id>/<run_id>/`. However, object names are also sometimes passed by
        upstream code as artifact paths relative to this root, rather than the full path. To keep upstream code simple,
        :class:`MLFlowObjectStore` accepts both relative MLflow artifact paths and absolute DBFS paths as object names.
        If an object name takes the form of
        `databricks/mlflow-tracking/<experiment_id>/<run_id>/artifacts/<artifact_path>`,
        it is assumed to be an absolute DBFS path, and the `<artifact_path>` is used when uploading objects to MLflow.
        Otherwise, the object name is assumed to be a relative MLflow artifact path, and the full provided name will be
        used as the artifact path when uploading to MLflow.

    Args:
        path (str): A DBFS path of the form
            `databricks/mlflow-tracking/<experiment_id>/<run_id>/artifacts/<path>`.
            `experiment_id` and `run_id` can be set as the format string placeholders
            `{mlflow_experiment_id}` and `{mlflow_run_id}'`.

            If both `experiment_id` and `run_id` are set as placeholders, the MLFlowObjectStore will be associated with
            the currently active MLflow run if one exists. If no active run exists, a new run will be created under a
            default experiment name, or the experiment name specified by the `MLFLOW_EXPERIMENT_NAME` environment
            variable if one is set.

            If `experiment_id` is provided and `run_id` is not, the MLFlowObjectStore will create a new run in the
            provided experiment.

            Providing a `run_id` without an `experiment_id` will raise an error.
        multipart_upload_chunk_size(int, optional): The maximum size of a single chunk in an MLflow multipart upload.
            The maximum number of chunks supported by MLflow is 10,000, so the max file size that can
            be uploaded is `10 000 * multipart_upload_chunk_size`. Defaults to 100MB for a max upload size of 1TB.
    """

    def __init__(self, path: str, multipart_upload_chunk_size: int = 100 * 1024 * 1024) -> None:
        try:
            import mlflow
            from mlflow import MlflowClient
        except ImportError as e:
            raise MissingConditionalImportError('mlflow', conda_package='mlflow>=2.9.2,<3.0') from e

        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as e:
            raise MissingConditionalImportError('databricks', conda_package='databricks-sdk>=0.15.0,<1.0') from e

        tracking_uri = os.getenv(
            mlflow.environment_variables.MLFLOW_TRACKING_URI.name,  # pyright: ignore[reportGeneralTypeIssues]
            MLFLOW_DATABRICKS_TRACKING_URI,
        )
        if tracking_uri != MLFLOW_DATABRICKS_TRACKING_URI:
            raise ValueError(
                'MLFlowObjectStore currently only supports Databricks-hosted MLflow tracking. '
                f'Environment variable `MLFLOW_TRACKING_URI` is set to a non-Databricks URI {tracking_uri}. '
                f'Please unset it or set the value to `{MLFLOW_DATABRICKS_TRACKING_URI}`.')

        # Use the Databricks WorkspaceClient to check that credentials are set up correctly.
        try:
            WorkspaceClient()
        except Exception as e:
            raise ValueError(
                f'Databricks SDK credentials not correctly setup. '
                'Visit https://databricks-sdk-py.readthedocs.io/en/latest/authentication.html#databricks-native-authentication '
                'to identify different ways to setup credentials.') from e

        self._mlflow_client = MlflowClient(tracking_uri)
        mlflow.environment_variables.MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.set(  # pyright: ignore[reportGeneralTypeIssues]
            multipart_upload_chunk_size,)

        experiment_id, run_id, _ = MLFlowObjectStore.parse_dbfs_path(path)
        if experiment_id == MLFLOW_EXPERIMENT_ID_PLACEHOLDER:
            experiment_id = None
        if run_id == MLFLOW_RUN_ID_PLACEHOLDER:
            run_id = None

        # Construct the `experiment_id` and `run_id` depending on whether format placeholders were provided.
        self.experiment_id, self.run_id = self._init_run_info(experiment_id, run_id)

    def _init_run_info(self, experiment_id: Optional[str], run_id: Optional[str]) -> Tuple[str, str]:
        """Returns the experiment ID and run ID for the MLflow run backing this object store.

        In a distributed setting, this should only be called on the rank 0 process.
        """
        import mlflow

        if experiment_id is None:
            if run_id is not None:
                raise ValueError('A `run_id` cannot be provided without a valid `experiment_id`.')

            active_run = mlflow.active_run()
            if active_run is not None:
                experiment_id = active_run.info.experiment_id
                run_id = active_run.info.run_id
                log.debug(f'MLFlowObjectStore using active MLflow run {run_id=}')
            else:
                # If no active run exists, create a new run for the default experiment.
                mlflow_env_var_name = mlflow.environment_variables.MLFLOW_EXPERIMENT_NAME.name  # pyright: ignore[reportGeneralTypeIssues]
                experiment_name = os.getenv(mlflow_env_var_name, DEFAULT_MLFLOW_EXPERIMENT_NAME)

                experiment = self._mlflow_client.get_experiment_by_name(experiment_name)
                if experiment is not None:
                    experiment_id = experiment.experiment_id
                else:
                    experiment_id = self._mlflow_client.create_experiment(experiment_name)

                run_id = self._mlflow_client.create_run(experiment_id).info.run_id

                log.debug(f'MLFlowObjectStore using a new MLflow run {run_id=}'
                          f'for new experiment "{experiment_name}" {experiment_id=}')
        else:
            if run_id is not None:
                # If a `run_id` is provided, check that it belongs to the provided experiment.
                run = self._mlflow_client.get_run(run_id)
                if run.info.experiment_id != experiment_id:
                    raise ValueError(
                        f'Provided `run_id` {run_id} does not belong to provided experiment {experiment_id}. '
                        f'Found experiment {run.info.experiment_id}.')

                log.debug(f'MLFlowObjectStore using provided MLflow run {run_id=} '
                          f'for provided experiment {experiment_id=}')
            else:
                # If no `run_id` is provided, create a new run in the provided experiment.
                run = self._mlflow_client.create_run(experiment_id)
                run_id = run.info.run_id

                log.debug(f'MLFlowObjectStore using new MLflow run {run_id=} '
                          f'for provided experiment {experiment_id=}')

        if experiment_id is None or run_id is None:
            raise ValueError('MLFlowObjectStore failed to initialize experiment and run ID.')

        return experiment_id, run_id

    @staticmethod
    def parse_dbfs_path(path: str) -> Tuple[str, str, str]:
        """Parses a DBFS path to extract the MLflow experiment ID, run ID, and relative artifact path.

        The path is expected to be of the format
        `databricks/mlflow-tracking/<experiment_id>/<run_id>/artifacts/<artifact_path>`.

        Args:
            path (str): The DBFS path to parse.

        Returns:
            (str, str, str): (experiment_id, run_id, artifact_path)

        Raises:
            ValueError: If the path is not of the expected format.
        """
        if not path.startswith(MLFLOW_DBFS_PATH_PREFIX):
            raise ValueError(f'DBFS MLflow path should start with {MLFLOW_DBFS_PATH_PREFIX}. Got: {path}')

        # Strip `databricks/mlflow-tracking/` and split into
        # `<experiment_id>`, `<run_id>`, `'artifacts'`, `<relative_path>``
        subpath = path[len(MLFLOW_DBFS_PATH_PREFIX):]
        mlflow_parts = subpath.split('/', maxsplit=3)

        if len(mlflow_parts) != 4 or mlflow_parts[2] != 'artifacts':
            raise ValueError(f'Databricks MLflow artifact path expected to be of the format '
                             f'{MLFLOW_DBFS_PATH_PREFIX}/<experiment_id>/<run_id>/artifacts/<relative_path>. '
                             f'Found {path=}')

        return mlflow_parts[0], mlflow_parts[1], mlflow_parts[3]

    def get_artifact_path(self, object_name: str) -> str:
        """Converts an object name into an MLflow relative artifact path.

        Args:
            object_name (str): The object name to convert. If the object name is a DBFS path beginning with
                ``MLFLOW_DBFS_PATH_PREFIX``, the path will be parsed to extract the MLflow relative artifact path.
                Otherwise, the object name is assumed to be a relative artifact path and will be returned as-is.
        """
        if object_name.startswith(MLFLOW_DBFS_PATH_PREFIX):
            experiment_id, run_id, object_name = self.parse_dbfs_path(object_name)
            if (experiment_id != self.experiment_id and experiment_id != MLFLOW_EXPERIMENT_ID_PLACEHOLDER):
                raise ValueError(f'Object {object_name} belongs to experiment ID {experiment_id}, '
                                 f'but MLFlowObjectStore is associated with experiment ID {self.experiment_id}.')
            if (run_id != self.run_id and run_id != MLFLOW_RUN_ID_PLACEHOLDER):
                raise ValueError(f'Object {object_name} belongs to run ID {run_id}, '
                                 f'but MLFlowObjectStore is associated with run ID {self.run_id}.')
        return object_name

    def get_dbfs_path(self, object_name: str) -> str:
        """Converts an object name to a full DBFS path."""
        artifact_path = self.get_artifact_path(object_name)
        return f'{MLFLOW_DBFS_PATH_PREFIX}{self.experiment_id}/{self.run_id}/artifacts/{artifact_path}'

    def get_uri(self, object_name: str) -> str:
        return 'dbfs:/' + self.get_dbfs_path(object_name)

    def upload_object(self,
                      object_name: str,
                      filename: Union[str, pathlib.Path],
                      callback: Optional[Callable[[int, int], None]] = None):
        del callback  # unused
        from mlflow.exceptions import MlflowException

        # Extract relative path from DBFS path.
        artifact_path = self.get_artifact_path(object_name)
        artifact_base_name = os.path.basename(artifact_path)
        artifact_dir = os.path.dirname(artifact_path)

        # Since MLflow doesn't support uploading artifacts with a different base name than the local file,
        # create a temporary symlink to the local file with the same base name as the desired artifact name.
        filename = os.path.abspath(filename)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_symlink_path = os.path.join(tmp_dir, artifact_base_name)
            os.symlink(filename, tmp_symlink_path)

            try:
                self._mlflow_client.log_artifact(self.run_id, tmp_symlink_path, artifact_dir)
            except MlflowException as e:
                _wrap_mlflow_exceptions(self.get_uri(object_name), e)

    def get_object_size(self, object_name: str) -> int:
        from mlflow.exceptions import MlflowException

        artifact = None
        try:
            artifact = self._get_artifact_info(object_name)
        except MlflowException as e:
            _wrap_mlflow_exceptions(self.get_uri(object_name), e)

        if artifact is not None:
            return artifact.file_size
        else:
            raise FileNotFoundError(f'Object {object_name} not found')

    def download_object(
        self,
        object_name: str,
        filename: Union[str, pathlib.Path],
        overwrite: bool = False,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        del callback  # unused
        from mlflow.exceptions import MlflowException

        # Since MlflowClient.download_artifacts only raises MlflowException with 500 Internal Error,
        # check for existence to surface a FileNotFoundError if necessary.
        artifact_path = self.get_artifact_path(object_name)
        artifact_info = self._get_artifact_info(object_name)
        if artifact_info is None:
            raise FileNotFoundError(f'Object {self.get_dbfs_path(artifact_path)} not found')

        filename = os.path.abspath(filename)
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(f'The file at {filename} already exists and overwrite is set to False.')

        # MLFLow doesn't support downloading artifacts directly to a specified filename, so instead
        # download to a temporary directory and then move the file to the desired location.
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                self._mlflow_client.download_artifacts(
                    run_id=self.run_id,
                    path=artifact_path,
                    dst_path=tmp_dir,
                )
                tmp_path = os.path.join(tmp_dir, artifact_path)

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                if overwrite:
                    os.replace(tmp_path, filename)
                else:
                    os.rename(tmp_path, filename)
            except MlflowException as e:
                _wrap_mlflow_exceptions(self.get_uri(artifact_path), e)

    def list_objects(self, prefix: Optional[str] = None) -> List[str]:
        """See :meth:`~composer.utils.ObjectStore.list_objects`.

        MLFlowObjectStore does not support listing objects with a prefix, so the ``prefix`` argument is ignored.
        """
        del prefix  # not supported for MLFlowObjectStore

        objects = []
        self._list_objects_helper(None, objects)
        return objects

    def _list_objects_helper(self, prefix: Optional[str], objects: List[str]) -> None:
        """Helper to recursively populate the full list of objects for ``list_objects``.

        Args:
            prefix (str | None): An artifact path prefix for artifacts to find.
            objects (list[str]): The list of DBFS object paths to populate.
        """
        from mlflow.exceptions import MlflowException

        artifact = None
        try:
            for artifact in self._mlflow_client.list_artifacts(self.run_id, prefix):
                if artifact.is_dir:
                    self._list_objects_helper(artifact.path, objects)
                else:
                    objects.append(artifact.path)
        except MlflowException as e:
            uri = '' if artifact is None else self.get_uri(artifact.path)
            _wrap_mlflow_exceptions(uri, e)

    def _get_artifact_info(self, object_name):
        """Get the :class:`~mlflow.entities.FileInfo` for the given object name.

        Args:
            object_name (str): The name of the object, either as an absolute DBFS path or a relative MLflow artifact path.

        Returns:
            Optional[FileInfo]: The :class:`~mlflow.entities.FileInfo` for the object, or None if it does not exist.
        """
        # MLflow doesn't support info for a singleton artifact, so we need to list all artifacts in the
        # parent path and find the one with the matching name.
        artifact_path = self.get_artifact_path(object_name)
        artifact_dir = os.path.dirname(artifact_path)
        artifacts = self._mlflow_client.list_artifacts(self.run_id, artifact_dir)
        for artifact in artifacts:
            if not artifact.is_dir and artifact.path == artifact_path:
                return artifact

        return None
