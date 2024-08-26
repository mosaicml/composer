# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log files to an object store."""

from __future__ import annotations

import logging
import multiprocessing
import os
import pathlib
import queue
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from multiprocessing.context import SpawnProcess
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from urllib.parse import urlparse

import torch

from composer.loggers import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import (
    MLFlowObjectStore,
    ObjectStore,
    ObjectStoreTransientError,
    build_remote_backend,
    dist,
    format_name_with_dist,
    get_file,
    retry,
    validate_credentials,
)
from composer.utils.object_store.mlflow_object_store import MLFLOW_DBFS_PATH_PREFIX

if TYPE_CHECKING:
    from composer.core import State
    from composer.devices import Device

log = logging.getLogger(__name__)

__all__ = ['RemoteUploaderDownloader']


class RemoteUploaderDownloader(LoggerDestination):
    r"""Logger destination that uploads (downloads) files to (from) a remote backend.

    This logger destination handles calls to :meth:`.Logger.upload_file`
    and uploads files to :class:`.ObjectStore`, such as AWS S3 or Google Cloud Storage. To minimize the training
    loop performance hit, it supports background uploads.

    .. testcode:: composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader.__init__

        from composer.loggers import RemoteUploaderDownloader
        from composer.utils import LibcloudObjectStore

        remote_uploader_downloader = RemoteUploaderDownloader(
            bucket_uri="s3://my-bucket",
        )

        # Construct the trainer using this logger
        trainer = Trainer(
            ...,
            loggers=[remote_uploader_downloader],
        )

    or

    .. testcode:: composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader.__init__

        from composer.loggers import RemoteUploaderDownloader
        from composer.utils import LibcloudObjectStore

        remote_uploader_downloader = RemoteUploaderDownloader(
            bucket_uri="libcloud://my-bucket",
            backend_kwargs={
                'provider': 's3',
                'container': 'my-bucket',
                'provider_kwargs': {
                    'key': 'AKIA...',
                    'secret': '*********',
                    'region': 'ap-northeast-1',
                },
            },
        )

        # Construct the trainer using this logger
        trainer = Trainer(
            ...,
            loggers=[remote_uploader_downloader],
        )

    or

    .. testcode:: composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader.__init__
        from composer.loggers import RemoteUploaderDownloader
        from composer.trainer import Trainer

        remote_uploader_downloader = RemoteUploaderDownloader(
            bucket_uri="libcloud://my-gcs-bucket",
            backend_kwargs={
                "provider": "google_storage",
                "container": "my-gcs-bucket",
                "key_environ": "MY_HMAC_ACCESS_ID", # Name of env variable for HMAC access id.
                "secret_environ": "MY_HMAC_SECRET", # Name of env variable for HMAC secret.
            },
        )

        # Construct the trainer using this logger
        trainer = Trainer(
            ...,
            loggers=[remote_uploader_downloader],
        )

    .. note::

        This callback blocks the training loop to upload each file, as
        the uploading happens in the background. Here are some additional tips for minimizing the performance impact:

        *   Set ``use_procs=True`` (the default) to use background processes, instead of threads, to perform the file
            uploads. Processes are recommended to ensure that the GIL is not blocking the training loop when
            performing CPU operations on uploaded files (e.g. computing and comparing checksums). Network I/O happens
            always occurs in the background.

        *   Provide a RAM disk path for the ``upload_staging_folder`` parameter. Copying files to stage on RAM will be
            faster than writing to disk. However, there must have sufficient excess RAM, or :exc:`MemoryError`\s may
            be raised.

    Args:
        bucket_uri (str): The remote uri for the bucket to use (e.g. s3://my-bucket).

            As individual :class:`.ObjectStore` instances are not necessarily thread safe, each worker will construct
            its own :class:`.ObjectStore` instance from ``remote_backend`` and ``backend_kwargs``.

        backend_kwargs (dict[str, Any]): The keyword arguments to construct the remote backend indicated by ``bucket_uri``.

            As individual :class:`.ObjectStore` instances are not necessarily thread safe, each worker will construct
            its own :class:`.ObjectStore` instance from ``remote_backend`` and ``backend_kwargs``.

        file_path_format_string (str, optional): A format string used to determine the remote file path (within the specified bucket).

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{remote_file_name}`` | The name of the file being logged.                    |
            +------------------------+-------------------------------------------------------+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`.State.run_name`.                              |
            +------------------------+-------------------------------------------------------+
            | ``{rank}``             | The global rank, as returned by                       |
            |                        | :func:`~composer.utils.dist.get_global_rank`.         |
            +------------------------+-------------------------------------------------------+
            | ``{local_rank}``       | The local rank of the process, as returned by         |
            |                        | :func:`~composer.utils.dist.get_local_rank`.          |
            +------------------------+-------------------------------------------------------+
            | ``{world_size}``       | The world size, as returned by                        |
            |                        | :func:`~composer.utils.dist.get_world_size`.          |
            +------------------------+-------------------------------------------------------+
            | ``{local_world_size}`` | The local world size, as returned by                  |
            |                        | :func:`~composer.utils.dist.get_local_world_size`.    |
            +------------------------+-------------------------------------------------------+
            | ``{node_rank}``        | The node rank, as returned by                         |
            |                        | :func:`~composer.utils.dist.get_node_rank`.           |
            +------------------------+-------------------------------------------------------+

            Leading slashes (``'/'``) will be stripped.

            Consider the following example, which subfolders the remote files by their rank:

            .. testsetup:: composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader.__init__.file_path_format_string

                import os

                os.makedirs('path/to', exist_ok=True)

                with open('path/to/file.txt', 'w+') as f:
                    f.write('hi')

            .. doctest:: composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader.__init__.file_path_format_string

                >>> remote_uploader_downloader = RemoteUploaderDownloader(..., file_path_format_string='rank_{rank}/{remote_file_name}')
                >>> trainer = Trainer(..., save_latest_filename=None, run_name='foo', loggers=[remote_uploader_downloader])
                >>> trainer.logger.upload_file(
                ...     remote_file_name='bar.txt',
                ...     file_path='path/to/file.txt',
                ... )

            .. testcleanup:: composer.loggers.remote_uploader_downloader.RemoteUploaderDownloader.__init__.file_path_format_string

                # Shut down the uploader
                remote_uploader_downloader._check_workers()
                remote_uploader_downloader.post_close()

            Assuming that the process's rank is ``0``, the remote backend would store the contents of
            ``'path/to/file.txt'`` in at ``'rank0/bar.txt'``.

            Default: ``'{remote_file_name}'``

        num_concurrent_uploads (int, optional): Maximum number of concurrent uploads. Defaults to 1.
        upload_staging_folder (str, optional): A folder to use for staging uploads.
            If not specified, defaults to using a :func:`~tempfile.TemporaryDirectory`.
        use_procs (bool, optional): Whether to perform file uploads in background processes (as opposed to threads).
            Defaults to True.
        num_attempts (int, optional): For operations that fail with a transient error, the number of attempts to make.
            Defaults to 3.
    """

    def __init__(
        self,
        bucket_uri: str,
        backend_kwargs: Optional[dict[str, Any]] = None,
        file_path_format_string: str = '{remote_file_name}',
        num_concurrent_uploads: int = 1,
        upload_staging_folder: Optional[str] = None,
        use_procs: bool = True,
        num_attempts: int = 3,
    ) -> None:
        parsed_remote_bucket = urlparse(bucket_uri)
        self.remote_backend_name, self.remote_bucket_name = parsed_remote_bucket.scheme, parsed_remote_bucket.netloc
        self.backend_kwargs = backend_kwargs if backend_kwargs is not None else {}
        if self.remote_backend_name in ['s3', 'oci', 'gs'] and 'bucket' not in self.backend_kwargs:
            self.backend_kwargs['bucket'] = self.remote_bucket_name
        elif self.remote_backend_name == 'sftp' and 'host' not in self.backend_kwargs:
            self.backend_kwargs['host'] = f'sftp://{self.remote_bucket_name}'
        elif self.remote_backend_name == 'libcloud' and 'container' not in self.backend_kwargs:
            self.backend_kwargs['container'] = self.remote_bucket_name

        self.file_path_format_string = file_path_format_string
        self.num_attempts = num_attempts
        self._run_name = None

        if upload_staging_folder is None:
            self._tempdir = tempfile.TemporaryDirectory()
            self._upload_staging_folder = self._tempdir.name
        else:
            self._tempdir = None
            self._upload_staging_folder = upload_staging_folder

        if num_concurrent_uploads < 1:
            raise ValueError('num_concurrent_uploads must be >= 1. Blocking uploads are not supported.')
        self._num_concurrent_uploads = num_concurrent_uploads

        # There could be multiple upload workers uploading to the same object
        # If multiple workers are uploading to the same object simultaneously (e.g. the checkpoint latest symlink file), then
        # The object store might keep the earlier file rather than the latter file as the "latest" version

        # To work around this, each object name can appear at most once in `self._file_upload_queue`
        # The main separately keeps track of {file_path_format_string: tempfile_path} for each API call to self.upload_file
        # and then periodically transfers items from this dictionary onto the file upload queue

        # Lock for modifying `logged_objects` or `enqueued_objects`
        # These objects are used by threads on the main process only
        self._object_lock = threading.Lock()

        # Files that were logged but yet to be enqueued. Mapping of the object name to the (tempfile path, overwrite) for that object
        self._logged_objects: dict[str, tuple[str, bool]] = {}

        # Set of enqueued objects. This should keep track of everything in self._file_upload_queue with O(1) lookup
        self._enqueued_objects: set[str] = set()

        # Thread that runs `self._enqueue_uploads`
        self._enqueue_thread = None
        # Event to signal the enqueue thread to shut down.
        self._enqueue_thread_flag = None

        if use_procs:
            mp_ctx = multiprocessing.get_context('spawn')
            self._file_upload_queue: Union[queue.Queue[tuple[str, str, bool]],
                                           multiprocessing.JoinableQueue[tuple[str, str, bool]],
                                          ] = mp_ctx.JoinableQueue()
            self._completed_queue: Union[queue.Queue[str], multiprocessing.JoinableQueue[str]] = mp_ctx.JoinableQueue()
            self._exception_queue: Union[queue.Queue[Exception],
                                         multiprocessing.JoinableQueue[Exception],
                                        ] = mp_ctx.JoinableQueue()
            self._finished_cls: Union[Callable[[],
                                               multiprocessing._EventType,  # pyright: ignore[reportGeneralTypeIssues]
                                              ],
                                      type[threading.Event],
                                     ] = mp_ctx.Event
            self._proc_class = mp_ctx.Process
        else:
            self._file_upload_queue = queue.Queue()
            self._completed_queue = queue.Queue()
            self._exception_queue = queue.Queue()
            self._finished_cls = threading.Event
            self._proc_class = threading.Thread
        self._worker_flag: Optional[
            Union[
                multiprocessing._EventType,  # pyright: ignore[reportGeneralTypeIssues]
                threading.Event,
            ]
        ] = None
        self._workers: list[Union[SpawnProcess, threading.Thread]] = []
        # the object store instance for the main thread. Deferring the construction of the object_store to first use.
        self._remote_backend = None

    @property
    def remote_backend(self) -> ObjectStore:
        """The :class:`.ObjectStore` instance for the main thread."""
        if self._remote_backend is None:
            self._remote_backend = build_remote_backend(self.remote_backend_name, self.backend_kwargs)
        return self._remote_backend

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused

        if self._worker_flag is not None:
            raise RuntimeError('The RemoteUploaderDownloader is already initialized.')
        self._worker_flag = self._finished_cls()
        self._run_name = state.run_name
        file_name_to_test = self._remote_file_name('.credentials_validated_successfully')

        # Create the enqueue thread
        self._enqueue_thread_flag = self._finished_cls()
        self._enqueue_thread = threading.Thread(target=self._enqueue_uploads, daemon=True)
        self._enqueue_thread.start()

        if dist.get_global_rank() == 0:
            retry(
                ObjectStoreTransientError,
                self.num_attempts,
            )(lambda: validate_credentials(self.remote_backend, file_name_to_test))()

        # If the remote backend is an `MLFlowObjectStore`, the original path kwarg may have placeholders that can be
        # updated with information generated at runtime, i.e., the MLFlow experiment and run IDs. This information
        # must be propagated across all ranks before the workers are started so that all workers use the same
        # MLFlow run.
        if self.backend_kwargs.get('path', '').startswith(MLFLOW_DBFS_PATH_PREFIX):
            if dist.get_global_rank() == 0:
                assert isinstance(self.remote_backend, MLFlowObjectStore)
                self.backend_kwargs['path'] = self.remote_backend.get_dbfs_path(self.backend_kwargs['path'])

            path_list = [self.backend_kwargs['path']]
            dist.broadcast_object_list(path_list, src=0)
            self.backend_kwargs['path'] = path_list[0]

        assert len(self._workers) == 0, 'workers should be empty if self._worker_flag was None'
        for _ in range(self._num_concurrent_uploads):
            worker = self._proc_class(
                target=_upload_worker,
                kwargs={
                    'file_queue': self._file_upload_queue,
                    'is_finished': self._worker_flag,
                    'remote_backend_name': self.remote_backend_name,
                    'backend_kwargs': self.backend_kwargs,
                    'num_attempts': self.num_attempts,
                    'completed_queue': self._completed_queue,
                    'exception_queue': self._exception_queue,
                },
                # The worker threads are joined in the shutdown procedure, so it is OK to set the daemon status.
                # Setting daemon status prevents the process from hanging if close was never called (e.g. in doctests)
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._check_workers()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._check_workers()

    @property
    def _all_workers_alive(self):
        """Whether all workers are alive."""
        return all(worker.is_alive() for worker in self._workers)

    def _check_workers(self):
        # Periodically check to see if any of the upload workers crashed
        # They would crash if:
        #   a) A file could not be uploaded, and the retry counter failed, or
        #   b) overwrite=False, but the file already exists,
        if not self._all_workers_alive:
            if not self._exception_queue.empty():
                exception = self._exception_queue.get_nowait()
                raise exception
            else:
                raise RuntimeError('Upload worker crashed. Please check the logs.')

    def upload_file(
        self,
        state: State,
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        copied_path = os.path.join(self._upload_staging_folder, str(uuid.uuid4()))
        os.makedirs(self._upload_staging_folder, exist_ok=True)
        shutil.copy2(file_path, copied_path)
        formatted_remote_file_name = self._remote_file_name(remote_file_name)
        with self._object_lock:
            if formatted_remote_file_name in self._logged_objects and not overwrite:
                raise FileExistsError(
                    f'Object {formatted_remote_file_name} was already enqueued to be uploaded, but overwrite=False.',
                )
            self._logged_objects[formatted_remote_file_name] = (copied_path, overwrite)

    def can_upload_files(self) -> bool:
        """Whether the logger supports uploading files."""
        return True

    def _enqueue_uploads(self):
        """Worker thread to enqueue uploads.

        This thread does two things:

        1.  It enqueues objects from ``self._logged_objects`` onto ``self._file_upload_queue``.
        2.  It keeps ``self._enqueued_objects`` in sync with ``self._file_upload_queue`` by listening to ``self._completed_uploads``.
        """
        assert self._enqueue_thread_flag is not None
        while True:
            with self._object_lock:
                # Remove all objects from self._enqueued_objects that have been successfully uploaded
                while True:
                    try:
                        object_name = self._completed_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._enqueued_objects.remove(object_name)
                    self._completed_queue.task_done()

                # Enqueue all objects that are in self._logged_objects but not in self._file_upload_queue
                objects_to_delete = []
                for object_name, (copied_path, overwrite) in self._logged_objects.items():
                    if object_name in self._enqueued_objects:
                        continue
                    self._file_upload_queue.put_nowait((copied_path, object_name, overwrite))
                    objects_to_delete.append(object_name)
                    self._enqueued_objects.add(object_name)
                for object_name in objects_to_delete:
                    del self._logged_objects[object_name]

                # Shutdown if the enqueue thread flag is set, which means that no more objects will be added to
                # self._logged_objects
                if self._enqueue_thread_flag.is_set():
                    if self._all_workers_alive:
                        if len(self._logged_objects) == 0:
                            # If finished (i.e. no more objects to be added to self._logged_objects) and all logged objects are
                            # enqueued, then break
                            break
                    else:
                        # If any worker died, then it's impossible to recover since the file was already popped off of the queue,
                        # so break. Some files may not be uploaded.
                        break

            time.sleep(0.2)  # Yield lock for `self.upload_file`

    def download_file(
        self,
        remote_file_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        get_file(
            path=remote_file_name,
            destination=destination,
            object_store=self.remote_backend,
            overwrite=overwrite,
            progress_bar=progress_bar,
        )

    def fit_end(self, state: State, logger: Logger):
        self.wait_for_workers(state.device)

    def eval_standalone_end(self, state: State, logger: Logger):
        self.wait_for_workers(state.device)

    def predict_end(self, state: State, logger: Logger):
        self.wait_for_workers(state.device)

    def wait_for_workers(self, device: Device):
        """Wait for all tasks to be completed.

        This is called after fit/eval/predict. If we don't wait, then a worker might not schedule
        an upload before the interpreter is shutdown and garbage collection begins. While
        post_close logic ensures existing uploads are completed, trying to schedule new uploads
        during this time will error.
        """
        # Verify enqueue thread has processed all tasks unless a worker threw an exception
        while self._exception_queue.empty():
            with self._object_lock:
                if len(self._logged_objects) == 0:
                    break
            time.sleep(0.2)  # Yield lock for enqueue thread

        # Verify all tasks have been completed unless a worker threw an exception
        all_ranks_upload_done_tensor = device.tensor_to_device(
            torch.tensor(
                [int(not self._file_upload_queue.empty() and self._exception_queue.empty())],
                dtype=torch.uint8,
            ),
        )
        dist.all_reduce(all_ranks_upload_done_tensor, reduce_operation='MAX')
        upload_not_done = all_ranks_upload_done_tensor.item() == 1
        while upload_not_done:
            time.sleep(2)
            all_ranks_upload_done_tensor = device.tensor_to_device(
                torch.tensor(
                    [int(not self._file_upload_queue.empty() and self._exception_queue.empty())],
                    dtype=torch.uint8,
                ),
            )
            dist.all_reduce(all_ranks_upload_done_tensor, reduce_operation='MAX')
            upload_not_done = all_ranks_upload_done_tensor.item() == 1

        if not self._exception_queue.empty():
            e = self._exception_queue.get_nowait()
            raise e

    def post_close(self):
        # Shutdown logic:
        # 1. Signal to the enqueue thread that all uploads are enqueued. Specifically.
        #    set a flag indicating that that no more objects will be added to self._logged_objects.
        # 2. Wait for the enqueue thread to shut down. It will only shut down once all objects are added to
        #    self._file_upload_queue. This will mean that self._logged_objects is empty.
        # 3. Send a flag to the workers that all uploads are enqueued in self._file_upload_queue.
        # 4. Wait for the workers to shut down. This means that all files have been uploaded
        if self._enqueue_thread_flag is not None:
            self._enqueue_thread_flag.set()

        if self._enqueue_thread is not None:
            self._enqueue_thread.join()

        if self._worker_flag is not None:
            self._worker_flag.set()

        # Then, ensure all workers have finished all uploads
        for worker in self._workers:
            worker.join()

        # Clean up the tempdir
        if self._tempdir is not None:
            self._tempdir.cleanup()

        # Empty the completed queue
        # This cleanup will not be done by the enqueue_thread anymore, as that thread has been shut down
        while True:
            try:
                object_name = self._completed_queue.get_nowait()
            except queue.Empty:
                break
            self._enqueued_objects.remove(object_name)
            self._completed_queue.task_done()

        if len(self._enqueued_objects) > 0 or len(self._logged_objects) > 0:
            # Warn on all objects that have not been uploaded
            object_names = list(self._enqueued_objects)
            object_names.extend(self._logged_objects.keys())
            warnings.warn(
                RuntimeWarning(
                    'The following objects may not have been uploaded, likely due to a worker crash: ' +
                    ', '.join(self._enqueued_objects),
                ),
            )

        # Reset all variables
        self._logged_objects.clear()
        self._enqueued_objects.clear()
        self._enqueue_thread = None
        self._tempdir = None
        self._worker_flag = None
        self._enqueue_thread_flag = None
        self._workers.clear()

    def get_uri_for_file(self, remote_file_name: str) -> str:
        """Get the object store provider uri for a remote file.

        Args:
            remote_file_name (str): The name of a remote file.

        Returns:
            str: The uri corresponding to the uploaded location of the remote file.
        """
        formatted_remote_file_name = self._remote_file_name(remote_file_name)
        return self.remote_backend.get_uri(formatted_remote_file_name.lstrip('/'))

    def _remote_file_name(self, remote_file_name: str):
        """Format the ``remote_file_name`` according to the ``file_path_format_string``."""
        if self._run_name is None:
            raise RuntimeError('The run name is not set. It should have been set on Event.INIT.')
        key_name = format_name_with_dist(
            self.file_path_format_string,
            run_name=self._run_name,
            remote_file_name=remote_file_name,
        )
        key_name = key_name.lstrip('/')

        return key_name


def _upload_worker(
    file_queue: Union[queue.Queue[tuple[str, str, bool]], multiprocessing.JoinableQueue[tuple[str, str, bool]]],
    completed_queue: Union[queue.Queue[str], multiprocessing.JoinableQueue[str]],
    exception_queue: Union[queue.Queue[Exception], multiprocessing.JoinableQueue[Exception]],
    is_finished: Union[multiprocessing._EventType, threading.Event],  # pyright: ignore[reportGeneralTypeIssues]
    remote_backend_name: str,
    backend_kwargs: dict[str, Any],
    num_attempts: int,
):
    """A long-running function to handle uploading files to the object store.

    The worker will continuously poll ``file_queue`` for files to upload. Once ``is_finished`` is set, the worker will
    exit once ``file_queue`` is empty.
    """
    remote_backend = build_remote_backend(remote_backend_name, backend_kwargs)
    while True:
        try:
            file_path_to_upload, remote_file_name, overwrite = file_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            if is_finished.is_set():
                break
            else:
                continue
        uri = remote_backend.get_uri(remote_file_name)

        # defining as a function-in-function to use decorator notation with num_attempts as an argument
        @retry(ObjectStoreTransientError, num_attempts=num_attempts)
        def upload_file(retry_index: int = 0):
            if retry_index == 0 and not overwrite:
                try:
                    remote_backend.get_object_size(remote_file_name)
                except FileNotFoundError:
                    # Good! It shouldn't exist.
                    pass
                else:
                    # Exceptions will be detected on the next batch_end or epoch_end event
                    e = FileExistsError(f'Object {uri} already exists, but overwrite was set to False.')
                    exception_queue.put_nowait(e)
                    raise e
            log.info('Uploading file %s to %s', file_path_to_upload, uri)
            try:
                remote_backend.upload_object(
                    object_name=remote_file_name,
                    filename=file_path_to_upload,
                )
            except Exception as e:
                exception_queue.put_nowait(e)
                raise e
            os.remove(file_path_to_upload)
            file_queue.task_done()
            completed_queue.put_nowait(remote_file_name)

        # When encountering issues with too much concurrency in uploads, staggering the uploads can help.
        # This stagger is intended for use when uploading model shards from every rank, and will effectively reduce
        # the concurrency by a factor of num GPUs per node.
        local_rank = dist.get_local_rank()
        local_rank_stagger = int(os.environ.get('COMPOSER_LOCAL_RANK_STAGGER_SECONDS', 0))
        log.debug(f'Staggering uploads by {local_rank * local_rank_stagger} seconds on {local_rank} local rank.')
        time.sleep(local_rank * local_rank_stagger)

        upload_file()
