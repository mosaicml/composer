# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log artifacts to an object store."""

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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import ObjectStore, ObjectStoreTransientError, dist, format_name_with_dist, get_file, retry

log = logging.getLogger(__name__)

__all__ = ['ObjectStoreLogger']


def _always_log(state: State, log_level: LogLevel, artifact_name: str):
    """Function that can be passed into ``should_log_artifact`` to log all artifacts."""
    del state, log_level, artifact_name  # unused
    return True


def _build_object_store(object_store_cls: Type[ObjectStore], object_store_kwargs: Dict[str, Any]):
    # error: Expected no arguments to "ObjectStore" constructor
    return object_store_cls(**object_store_kwargs)  # type: ignore


class ObjectStoreLogger(LoggerDestination):
    r"""Logger destination that uploads artifacts to an object store.

    This logger destination handles calls to :meth:`.Logger.file_artifact`
    and uploads files to :class:`.ObjectStore`, such as AWS S3 or Google Cloud Storage. To minimize the training
    loop performance hit, it supports background uploads.

    .. testcode:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__

        from composer.loggers import ObjectStoreLogger
        from composer.utils import LibcloudObjectStore

        object_store_logger = ObjectStoreLogger(
            object_store_cls=LibcloudObjectStore,
            object_store_kwargs={
                'provider': 's3',
                'container': 'my-bucket',
                'provider_kwargs=': {
                    'key': 'AKIA...',
                    'secret': '*********',
                    'region': 'ap-northeast-1',
                },
            },
        )

        # Construct the trainer using this logger
        trainer = Trainer(
            ...,
            loggers=[object_store_logger],
        )

    .. note::

        This callback blocks the training loop to copy each artifact where ``should_log_artifact`` returns ``True``, as
        the uploading happens in the background. Here are some additional tips for minimizing the performance impact:

        *   Set ``should_log`` to filter which artifacts will be logged. By default, all artifacts are logged.

        *   Set ``use_procs=True`` (the default) to use background processes, instead of threads, to perform the file
            uploads. Processes are recommended to ensure that the GIL is not blocking the training loop when
            performing CPU operations on uploaded files (e.g. computing and comparing checksums). Network I/O happens
            always occurs in the background.

        *   Provide a RAM disk path for the ``upload_staging_folder`` parameter. Copying files to stage on RAM will be
            faster than writing to disk. However, there must have sufficient excess RAM, or :exc:`MemoryError`\s may
            be raised.

    Args:
        object_store_cls (Type[ObjectStore]): The object store class.

            As individual :class:`.ObjectStore` instances are not necessarily thread safe, each worker will construct
            its own :class:`.ObjectStore` instance from ``object_store_cls`` and ``object_store_kwargs``.

        object_store_kwargs (Dict[str, Any]): The keyword arguments to construct ``object_store_cls``.

            As individual :class:`.ObjectStore` instances are not necessarily thread safe, each worker will construct
            its own :class:`.ObjectStore` instance from ``object_store_cls`` and ``object_store_kwargs``.

        should_log_artifact ((State, LogLevel, str) -> bool, optional): A function to filter which artifacts
            are uploaded.

            The function should take the (current training state, log level, artifact name) and return a boolean
            indicating whether this file should be uploaded.

            By default, all artifacts will be uploaded.

        object_name (str, optional): A format string used to determine the object name.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{artifact_name}``    | The name of the artifact being logged.                |
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

            Consider the following example, which subfolders the artifacts by their rank:

            .. testsetup:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__.object_name

                import os

                os.makedirs('path/to', exist_ok=True)

                with open('path/to/file.txt', 'w+') as f:
                    f.write('hi')

            .. doctest:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__.object_name

                >>> object_store_logger = ObjectStoreLogger(..., object_name='rank_{rank}/{artifact_name}')
                >>> trainer = Trainer(..., run_name='foo', loggers=[object_store_logger])
                >>> trainer.logger.file_artifact(
                ...     log_level=LogLevel.EPOCH,
                ...     artifact_name='bar.txt',
                ...     file_path='path/to/file.txt',
                ... )

            .. testcleanup:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__.object_name

                # Shut down the uploader
                object_store_logger._check_workers()
                object_store_logger.post_close()

            Assuming that the process's rank is ``0``, the object store would store the contents of
            ``'path/to/file.txt'`` in an object named ``'rank0/bar.txt'``.

            Default: ``'{artifact_name}'``

        num_concurrent_uploads (int, optional): Maximum number of concurrent uploads. Defaults to 4.
        upload_staging_folder (str, optional): A folder to use for staging uploads.
            If not specified, defaults to using a :func:`~tempfile.TemporaryDirectory`.
        use_procs (bool, optional): Whether to perform file uploads in background processes (as opposed to threads).
            Defaults to True.
        num_attempts (int, optional): For operations that fail with a transient error, the number of attempts to make.
            Defaults to 3.
    """

    def __init__(self,
                 object_store_cls: Type[ObjectStore],
                 object_store_kwargs: Dict[str, Any],
                 should_log_artifact: Optional[Callable[[State, LogLevel, str], bool]] = None,
                 object_name: str = '{artifact_name}',
                 num_concurrent_uploads: int = 4,
                 upload_staging_folder: Optional[str] = None,
                 use_procs: bool = True,
                 num_attempts: int = 3) -> None:
        self.object_store_cls = object_store_cls
        self.object_store_kwargs = object_store_kwargs
        if should_log_artifact is None:
            should_log_artifact = _always_log
        self.should_log_artifact = should_log_artifact
        self.object_name = object_name
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
        # The main separately keeps track of {object_name: tempfile_path} for each API call to self.log_file_artifact
        # and then periodically transfers items from this dictionary onto the file upload queue

        # Lock for modifying `logged_objects` or `enqueued_objects`
        # These objects are used by threads on the main process only
        self._object_lock = threading.Lock()

        # Files that were logged but yet to be enqueued. Mapping of the object name to the (tempfile path, overwrite) for that object
        self._logged_objects: Dict[str, Tuple[str, bool]] = {}

        # Set of enqueued objects. This should keep track of everything in self._file_upload_queue with O(1) lookup
        self._enqueued_objects: Set[str] = set()

        # Thread that runs `self._enqueue_uploads`
        self._enqueue_thread = None
        # Event to signal the enqueue thread to shut down.
        self._enqueue_thread_flag = None

        if use_procs:
            mp_ctx = multiprocessing.get_context('spawn')
            self._file_upload_queue: Union[queue.Queue[Tuple[str, str, bool]],
                                           multiprocessing.JoinableQueue[Tuple[str, str,
                                                                               bool]],] = mp_ctx.JoinableQueue()
            self._completed_queue: Union[queue.Queue[str], multiprocessing.JoinableQueue[str],] = mp_ctx.JoinableQueue()
            self._finished_cls: Union[Callable[[], multiprocessing._EventType], Type[threading.Event]] = mp_ctx.Event
            self._proc_class = mp_ctx.Process
        else:
            self._file_upload_queue = queue.Queue()
            self._completed_queue = queue.Queue()
            self._finished_cls = threading.Event
            self._proc_class = threading.Thread
        self._worker_flag: Optional[Union[multiprocessing._EventType, threading.Event]] = None
        self._workers: List[Union[SpawnProcess, threading.Thread]] = []
        # the object store instance for the main thread. Deferring the construction of the object_store to first use.
        self._object_store = None

    @property
    def object_store(self) -> ObjectStore:
        """The :class:`.ObjectStore` instance for the main thread."""
        if self._object_store is None:
            self._object_store = _build_object_store(self.object_store_cls, self.object_store_kwargs)
        return self._object_store

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        if self._worker_flag is not None:
            raise RuntimeError('The ObjectStoreLogger is already initialized.')
        self._worker_flag = self._finished_cls()
        self._run_name = state.run_name
        object_name_to_test = self._object_name('.credentials_validated_successfully')

        # Create the enqueue thread
        self._enqueue_thread_flag = self._finished_cls()
        self._enqueue_thread = threading.Thread(target=self._enqueue_uploads, daemon=True)
        self._enqueue_thread.start()

        if dist.get_global_rank() == 0:
            retry(ObjectStoreTransientError,
                  self.num_attempts)(lambda: _validate_credentials(self.object_store, object_name_to_test))()
        assert len(self._workers) == 0, 'workers should be empty if self._worker_flag was None'
        for _ in range(self._num_concurrent_uploads):
            worker = self._proc_class(
                target=_upload_worker,
                kwargs={
                    'file_queue': self._file_upload_queue,
                    'is_finished': self._worker_flag,
                    'object_store_cls': self.object_store_cls,
                    'object_store_kwargs': self.object_store_kwargs,
                    'num_attempts': self.num_attempts,
                    'completed_queue': self._completed_queue,
                },
                # The worker threads are joined in the shutdown procedure, so it is OK to set the daemon status
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
        #   b) allow_overwrite=False, but the file already exists,
        if not self._all_workers_alive:
            raise RuntimeError('Upload worker crashed. Please check the logs.')

    def log_file_artifact(
        self,
        state: State,
        log_level: LogLevel,
        artifact_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool,
    ):
        if not self.should_log_artifact(state, log_level, artifact_name):
            return
        copied_path = os.path.join(self._upload_staging_folder, str(uuid.uuid4()))
        os.makedirs(self._upload_staging_folder, exist_ok=True)
        shutil.copy2(file_path, copied_path)
        object_name = self._object_name(artifact_name)
        with self._object_lock:
            if object_name in self._logged_objects and not overwrite:
                raise FileExistsError(f'Object {object_name} was already enqueued to be uploaded, but overwrite=False.')
            self._logged_objects[object_name] = (copied_path, overwrite)

    def can_log_file_artifacts(self) -> bool:
        """Whether the logger supports logging file artifacts."""
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
                        # so break.Some files may not be uploaded.
                        break

            # Yield the lock, so it can be acquired by `self.log_file_artifact`
            time.sleep(0.2)

    def get_file_artifact(
        self,
        artifact_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        get_file(path=artifact_name,
                 destination=destination,
                 object_store=self.object_store,
                 overwrite=overwrite,
                 progress_bar=progress_bar)

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
                RuntimeWarning('The following objects may not have been uploaded, likely due to a worker crash: ' +
                               ', '.join(self._enqueued_objects)))

        # Reset all variables
        self._logged_objects.clear()
        self._enqueued_objects.clear()
        self._enqueue_thread = None
        self._tempdir = None
        self._worker_flag = None
        self._enqueue_thread_flag = None
        self._workers.clear()

    def get_uri_for_artifact(self, artifact_name: str) -> str:
        """Get the object store provider uri for an artfact.

        Args:
            artifact_name (str): The name of an artifact.

        Returns:
            str: The uri corresponding to the uploaded location of the artifact.
        """
        obj_name = self._object_name(artifact_name)
        return self.object_store.get_uri(obj_name.lstrip('/'))

    def _object_name(self, artifact_name: str):
        """Format the ``artifact_name`` according to the ``object_name_string``."""
        if self._run_name is None:
            raise RuntimeError('The run name is not set. It should have been set on Event.INIT.')
        key_name = format_name_with_dist(
            self.object_name,
            run_name=self._run_name,
            artifact_name=artifact_name,
        )
        key_name = key_name.lstrip('/')

        return key_name


def _validate_credentials(
    object_store: ObjectStore,
    object_name_to_test: str,
) -> None:
    # Validates the credentials by attempting to touch a file in the bucket
    # raises an error if there was a credentials failure.
    with tempfile.NamedTemporaryFile('wb') as f:
        f.write(b'credentials_validated_successfully')
        object_store.upload_object(
            object_name=object_name_to_test,
            filename=f.name,
        )


def _upload_worker(
    file_queue: Union[queue.Queue[Tuple[str, str, bool]], multiprocessing.JoinableQueue[Tuple[str, str, bool]]],
    completed_queue: Union[queue.Queue[str], multiprocessing.JoinableQueue[str]],
    is_finished: Union[multiprocessing._EventType, threading.Event],
    object_store_cls: Type[ObjectStore],
    object_store_kwargs: Dict[str, Any],
    num_attempts: int,
):
    """A long-running function to handle uploading files to the object store.

    The worker will continuously poll ``file_queue`` for files to upload. Once ``is_finished`` is set, the worker will
    exit once ``file_queue`` is empty.
    """
    object_store = _build_object_store(object_store_cls, object_store_kwargs)
    while True:
        try:
            file_path_to_upload, object_name, overwrite = file_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            if is_finished.is_set():
                break
            else:
                continue
        uri = object_store.get_uri(object_name)

        # defining as a function-in-function to use decorator notation with num_attempts as an argument
        @retry(ObjectStoreTransientError, num_attempts=num_attempts)
        def upload_file():
            if not overwrite:
                try:
                    object_store.get_object_size(object_name)
                except FileNotFoundError:
                    # Good! It shouldn't exist.
                    pass
                else:
                    # Exceptions will be detected on the next batch_end or epoch_end event
                    raise FileExistsError(f'Object {uri} already exists, but allow_overwrite was set to False.')
            log.info('Uploading file %s to %s', file_path_to_upload, uri)
            object_store.upload_object(
                object_name=object_name,
                filename=file_path_to_upload,
            )
            os.remove(file_path_to_upload)
            file_queue.task_done()
            completed_queue.put_nowait(object_name)

        upload_file()
