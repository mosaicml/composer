# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Util class to upload file."""

import logging
import multiprocessing
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from enum import Enum
from typing import Any, Optional

from composer.utils.dist import broadcast_object_list, get_global_rank, get_local_rank
from composer.utils.file_helpers import (
    parse_uri,
    validate_credentials,
)
from composer.utils.object_store.mlflow_object_store import MLFLOW_DBFS_PATH_PREFIX, MLFlowObjectStore
from composer.utils.object_store.object_store import (
    ObjectStore,
    ObjectStoreTransientError,
)
from composer.utils.object_store.utils import build_remote_backend
from composer.utils.retrying import retry

log = logging.getLogger(__name__)

__all__ = ['RemoteUploader']


class RemoteFilesExistingCheckStatus(Enum):
    EXIST = 1
    TIMEOUT = 2
    ERROR = 3


def _check_remote_files_exists(
    remote_backend_name: str,
    backend_kwargs: dict[str, Any],
    remote_checkpoint_file_names: list[str],
    main_process_pid: int,
    is_remote_upload_failed: multiprocessing.Event, # pyright: ignore[reportGeneralTypeIssues]
    max_wait_time_in_seconds: int = 3600,
    wait_before_next_try_in_seconds: float = 30,
):
    start_time = time.time()
    object_store = build_remote_backend(remote_backend_name, backend_kwargs)

    for remote_file_name in remote_checkpoint_file_names:
        while True:
            if is_remote_upload_failed.is_set():
                log.debug(f'Stop symlink uploading since the checkpoint files uploading failed')
                return RemoteFilesExistingCheckStatus.ERROR
            # Return if parent process exits
            try:
                os.kill(main_process_pid, 0)
            except OSError:
                return RemoteFilesExistingCheckStatus.ERROR
            try:
                object_store.get_object_size(remote_file_name)
                break
            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    log.debug(f'Got exception {type(e)}: {str(e)} when accessing remote file {remote_file_name}')
                time.sleep(wait_before_next_try_in_seconds)
            if time.time() - start_time > max_wait_time_in_seconds:
                return RemoteFilesExistingCheckStatus.TIMEOUT
    return RemoteFilesExistingCheckStatus.EXIST


def _upload_file_to_object_store(
    remote_backend_name: str,
    backend_kwargs: dict[str, Any],
    remote_file_name: str,
    local_file_path: str,
    overwrite: bool,
    num_attempts: int,
) -> int:
    object_store = build_remote_backend(remote_backend_name, backend_kwargs)

    @retry(ObjectStoreTransientError, num_attempts=num_attempts)
    def upload_file(retry_index: int = 0):
        if retry_index == 0 and not overwrite:
            try:
                object_store.get_object_size(remote_file_name)
            except FileNotFoundError:
                # Good! It shouldn't exist.
                pass
            else:
                raise FileExistsError(
                    f'Object {remote_file_name} already exists, but overwrite was set to False. '
                    'Please set `save_overwrite` to `True` in Trainer to overwrite the existing file.',
                )
        log.info(f'Uploading file {local_file_path} to {remote_file_name}')
        object_store.upload_object(
            object_name=remote_file_name,
            filename=local_file_path,
        )
        os.remove(local_file_path)

    log.info(f'Finished uploading file {local_file_path} to {remote_file_name}')
    # When encountering issues with too much concurrency in uploads, staggering the uploads can help.
    # This stagger is intended for use when uploading model shards from every rank, and will effectively reduce
    # the concurrency by a factor of num GPUs per node.
    local_rank = get_local_rank()
    local_rank_stagger = int(os.environ.get('COMPOSER_LOCAL_RANK_STAGGER_SECONDS', 0))
    log.debug(f'Staggering uploads by {local_rank * local_rank_stagger} seconds on {local_rank} local rank.')
    time.sleep(local_rank * local_rank_stagger)
    upload_file()
    return 0


class RemoteUploader:
    """Class for uploading a file to object store asynchronously."""

    def __init__(
        self,
        remote_folder: str,
        backend_kwargs: Optional[dict[str, Any]] = None,
        num_concurrent_uploads: int = 2,
        num_attempts: int = 3,
    ):
        if num_concurrent_uploads < 1 or num_attempts < 1:
            raise ValueError(
                f'num_concurrent_uploads and num_attempts must be >= 1, but got {num_concurrent_uploads} and {num_attempts}',
            )

        self.remote_folder = remote_folder
        # A folder to use for staging uploads
        self._tempdir = tempfile.TemporaryDirectory()
        self._upload_staging_folder = self._tempdir.name
        self.remote_backend_name, self.remote_bucket_name, self.path = parse_uri(remote_folder)

        self.backend_kwargs: dict[str, Any] = backend_kwargs if backend_kwargs is not None else {}
        if self.remote_backend_name in ['s3', 'oci', 'gs'] and 'bucket' not in self.backend_kwargs:
            self.backend_kwargs['bucket'] = self.remote_bucket_name
        elif self.remote_backend_name == 'libcloud':
            if 'container' not in self.backend_kwargs:
                self.backend_kwargs['container'] = self.remote_bucket_name
        elif self.remote_backend_name == 'azure':
            self.remote_backend_name = 'libcloud'
            self.backend_kwargs = {
                'provider': 'AZURE_BLOBS',
                'container': self.remote_bucket_name,
                'key_environ': 'AZURE_ACCOUNT_NAME',
                'secret_environ': 'AZURE_ACCOUNT_ACCESS_KEY',
            }
        elif self.remote_backend_name == 'dbfs':
            self.backend_kwargs['path'] = self.path
        elif self.remote_backend_name == 'wandb':
            raise NotImplementedError(
                f'There is no implementation for WandB via URI. Please use '
                'WandBLogger with log_artifacts set to True.',
            )
        else:
            raise NotImplementedError(
                f'There is no implementation for the cloud backend {self.remote_backend_name} via URI. Please use '
                'one of the supported object stores (s3, oci, gs, azure, dbfs).',
            )

        self.num_attempts = num_attempts
        self._remote_backend: Optional[ObjectStore] = None
        mp_context = multiprocessing.get_context('spawn')
        self.upload_executor = ProcessPoolExecutor(
            max_workers=num_concurrent_uploads,
            mp_context=mp_context,
        )
        self.check_remote_files_exist_executor = ProcessPoolExecutor(
            max_workers=2,
            mp_context=mp_context,
        )
        self.is_remote_upload_failed = mp_context.Manager().Event()

        # Used internally to track the future status.
        # If a future completed successfully, we'll remove it from this list
        # when check_workers() or wait() is called
        self.futures: list[Future] = []

        self.pid = os.getpid()

    @property
    def remote_backend(self) -> ObjectStore:
        if self._remote_backend is None:
            self._remote_backend = build_remote_backend(self.remote_backend_name, self.backend_kwargs)
        return self._remote_backend

    def init(self):
        # If it's dbfs path like: dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/
        # We need to fill out the experiment_id and run_id

        if get_global_rank() == 0:

            @retry(ObjectStoreTransientError, num_attempts=self.num_attempts)
            def _validate_credential_with_retry():
                validate_credentials(self.remote_backend, '.credentials_validated_successfully')

            _validate_credential_with_retry()
        if self.path.startswith(MLFLOW_DBFS_PATH_PREFIX):
            if get_global_rank() == 0:
                assert isinstance(self.remote_backend, MLFlowObjectStore)
                self.path = self.remote_backend.get_dbfs_path(self.path)
            path_list = [self.path]
            broadcast_object_list(path_list, src=0)
            self.path = path_list[0]
            self.backend_kwargs['path'] = self.path

    def upload_file_async(
        self,
        remote_file_name: str,
        file_path: pathlib.Path,
        overwrite: bool,
    ):
        """Async call to submit a job for uploading.

        It returns a future, so users can track the status of the individual future.
        User can also call wait() to wait for all the futures.
        """
        # Copy file to staging folder
        copied_path = os.path.join(self._upload_staging_folder, str(uuid.uuid4()))
        os.makedirs(self._upload_staging_folder, exist_ok=True)
        shutil.copy2(file_path, copied_path)

        # Async upload file
        future = self.upload_executor.submit(
            _upload_file_to_object_store,
            remote_backend_name=self.remote_backend_name,
            backend_kwargs=self.backend_kwargs,
            remote_file_name=remote_file_name,
            local_file_path=copied_path,
            overwrite=overwrite,
            num_attempts=self.num_attempts,
        )
        self.futures.append(future)
        return future

    def check_workers(self):
        """Non-blocking call to check workers are either running or done.

        Traverse self.futures, and check if it's completed
        1. if it completed with exception, raise that exception
        2. if it completed without exception, remove it from self.futures
        """
        done_futures: list[Future] = []
        for future in self.futures:
            if future.done():
                # future.exception is a blocking call
                exception_or_none = future.exception()
                if exception_or_none is not None:
                    self.is_remote_upload_failed.set()
                    raise exception_or_none
                else:
                    done_futures.append(future)
        for future in done_futures:
            self.futures.remove(future)

    def wait(self):
        """Blocking call to wait all the futures to complete.

        If a future is done successfully, remove it from self.futures(),
        otherwise, raise the exception
        """
        for future in self.futures:
            exception_or_none = future.exception()
            if exception_or_none is not None:
                self.is_remote_upload_failed.set()
                raise exception_or_none
        self.futures = []

    def wait_and_close(self):
        """Blocking call to wait all uploading to finish and close this uploader.

        After this function is called, users can not use this uploader
        to uploading anymore. So please only call wait_and_close() after submitting
        all uploading requests.
        """
        # make sure all workers are either running, or completed successfully
        self.wait()
        self.upload_executor.shutdown(wait=True)
        self.check_remote_files_exist_executor.shutdown(wait=True)
        log.debug('Finished all uploading tasks, closing RemoteUploader')

    def check_remote_files_exist_async(
        self,
        remote_checkpoint_file_names: list[str],
        max_wait_time_in_seconds: int = 3600,
        wait_before_next_try_in_seconds: float = 30,
    ):
        future = self.check_remote_files_exist_executor.submit(
            _check_remote_files_exists,
            remote_backend_name=self.remote_backend_name,
            backend_kwargs=self.backend_kwargs,
            remote_checkpoint_file_names=remote_checkpoint_file_names,
            main_process_pid=self.pid,
            is_remote_upload_failed=self.is_remote_upload_failed,
            max_wait_time_in_seconds=max_wait_time_in_seconds,
            wait_before_next_try_in_seconds=wait_before_next_try_in_seconds,
        )
        self.futures.append(future)
        return future
