# Copyright 2021 MosaicML. All Rights Reserved.

"""Log artifacts to an object store."""

from __future__ import annotations

import logging
import multiprocessing
import os
import pathlib
import queue
import shutil
import tempfile
import textwrap
import threading
import time
import uuid
from multiprocessing.context import SpawnProcess
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from libcloud.common.types import LibcloudError
from libcloud.storage.types import ObjectDoesNotExistError
from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.object_store import ObjectStore

log = logging.getLogger(__name__)

__all__ = ["ObjectStoreLogger"]


def _always_log(state: State, log_level: LogLevel, artifact_name: str):
    """Function that can be passed into ``should_log_artifact`` to log all artifacts."""
    del state, log_level, artifact_name  # unused
    return True


class ObjectStoreLogger(LoggerDestination):
    """Logger destination that uploads artifacts to an object store.

    This logger destination handles calls to :meth:`~composer.core.logging.Logger.file_artifact`
    and uploads files to an object store, such as AWS S3 or Google Cloud Storage.

    Example
        .. testsetup:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__

            import os
            import functools
            import composer.loggers.object_store_logger

            from composer.loggers import ObjectStoreLogger

            # For this example, we do not validate credentials
            def do_not_validate(*args, **kwargs) -> None:
                pass

            composer.loggers.object_store_logger._validate_credentials = do_not_validate
           
            ObjectStoreLogger = functools.partial(
                ObjectStoreLogger,
                use_procs=False,
                num_concurrent_uploads=1,
            )

        .. testcode:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__

            object_store_logger = ObjectStoreLogger(
                provider='s3',
                container='my-bucket',
                provider_kwargs={
                    'key': 'AKIA...',
                    'secret': '*********',
                    'region': 'ap-northeast-1',
                },
            )
            
            # Construct the trainer using this logger
            trainer = Trainer(
                ...,
                loggers=[object_store_logger],
            )

        .. testcleanup:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__

            # Shut down the uploader
            object_store_logger._check_workers()
            object_store_logger.post_close()

    .. note::

        This callback blocks the training loop to copy each artifact where ``should_log_artifact`` returns ``True``, as
        the uploading happens in the background. Here are some additional tips for minimizing the performance impact:

        *   Set ``should_log`` to filter which artifacts will be logged. By default, all artifacts are logged.

        *   Set ``use_procs=True`` (the default) to use background processes, instead of threads, to perform the file
            uploads. Processes are recommended to ensure that the GIL is not blocking the training loop when
            performing CPU operations on uploaded files (e.g. computing and comparing checksums). Network I/O happens
            always occurs in the background.

        *   Provide a RAM disk path for the ``upload_staging_folder`` parameter. Copying files to stage on RAM will be
            faster than writing to disk. However, there must have sufficient excess RAM, or :exc:`MemoryError`\\s may
            be raised.

    Args:
        provider (str): Cloud provider to use. Valid options are:

            * :mod:`~libcloud.storage.drivers.atmos`
            * :mod:`~libcloud.storage.drivers.auroraobjects`
            * :mod:`~libcloud.storage.drivers.azure_blobs`
            * :mod:`~libcloud.storage.drivers.backblaze_b2`
            * :mod:`~libcloud.storage.drivers.cloudfiles`
            * :mod:`~libcloud.storage.drivers.digitalocean_spaces`
            * :mod:`~libcloud.storage.drivers.google_storage`
            * :mod:`~libcloud.storage.drivers.ktucloud`
            * :mod:`~libcloud.storage.drivers.local`
            * :mod:`~libcloud.storage.drivers.minio`
            * :mod:`~libcloud.storage.drivers.nimbus`
            * :mod:`~libcloud.storage.drivers.ninefold`
            * :mod:`~libcloud.storage.drivers.oss`
            * :mod:`~libcloud.storage.drivers.rgw`
            * :mod:`~libcloud.storage.drivers.s3`

            .. seealso:: :doc:`Full list of libcloud providers <libcloud:storage/supported_providers>`

        container (str): The name of the container (i.e. bucket) to use.
        provider_kwargs (Dict[str, Any], optional):  Keyword arguments to pass into the constructor
            for the specified provider. These arguments would usually include the cloud region
            and credentials.
            
            Common keys are:

            * ``key`` (str): API key or username to be used (required).
            * ``secret`` (str): Secret password to be used (required).
            * ``secure`` (bool): Whether to use HTTPS or HTTP. Note: Some providers only support HTTPS, and it is on by default.
            * ``host`` (str): Override hostname used for connections.
            * ``port`` (int): Override port used for connections.
            * ``api_version`` (str): Optional API version. Only used by drivers which support multiple API versions.
            * ``region`` (str): Optional driver region. Only used by drivers which support multiple regions.

            .. seealso:: :class:`libcloud.storage.base.StorageDriver`

        should_log_artifact ((State, LogLevel, str) -> bool, optional): A function to filter which artifacts
            are uploaded.

            The function should take the (current training state, log level, artifact name) and return a boolean
            indicating whether this file should be uploaded.

            By default, all artifacts will be uploaded.

        object_name_format (str, optional): A format string used to determine the object name.

            The following format variables are available:

            +------------------------+-------------------------------------------------------+
            | Variable               | Description                                           |
            +========================+=======================================================+
            | ``{artifact_name}``    | The name of the artifact being logged.                |
            +------------------------+-------------------------------------------------------+
            | ``{run_name}``         | The name of the training run. See                     |
            |                        | :attr:`~composer.core.logging.Logger.run_name`.       |
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

            .. testsetup:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__.object_name_format

                import os
                import functools
                import composer.loggers.object_store_logger

                from composer.loggers import ObjectStoreLogger as OriginalObjectStoreLogger

                # For this example, we do not validate credentials
                def do_not_validate(*args, **kwargs) -> None:
                    pass

                composer.loggers.object_store_logger._validate_credentials = do_not_validate

                OriginalObjectStoreLogger.log_file_artifact = lambda *args, **kwargs: None

                def ObjectStoreLogger(fake_ellipsis = ..., *args, **kwargs):
                    return OriginalObjectStoreLogger(
                        use_procs=False,
                        num_concurrent_uploads=1,
                        provider='local',
                        container='.',
                        provider_kwargs={'key': os.path.abspath(".") },
                    )

            .. doctest:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__.object_name_format

                >>> object_store_logger = ObjectStoreLogger(..., object_name_format='rank_{rank}/{artifact_name}')
                >>> trainer = Trainer(..., run_name='foo', loggers=[object_store_logger])
                >>> trainer.logger.file_artifact(log_level=LogLevel.EPOCH, artifact_name='bar.txt',
                ... file_path='path/to/file.txt')

            .. testcleanup:: composer.loggers.object_store_logger.ObjectStoreLogger.__init__.object_name_format

                # Shut down the uploader
                object_store_logger._check_workers()
                object_store_logger.post_close()
           
            Assuming that the process's rank is ``0``, the object store would store the contents of
            ``'path/to/file.txt'`` in an object named ``'rank_0/bar.txt'``.

            Default: ``'{artifact_name}'``

        num_concurrent_uploads (int, optional): Maximum number of concurrent uploads. Defaults to 4.
        upload_staging_folder (str, optional): A folder to use for staging uploads.
            If not specified, defaults to using a :func:`~tempfile.TemporaryDirectory`.
        use_procs (bool, optional): Whether to perform file uploads in background processes (as opposed to threads).
            Defaults to True.
    """

    def __init__(
        self,
        provider: str,
        container: str,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        should_log_artifact: Optional[Callable[[State, LogLevel, str], bool]] = None,
        object_name_format: str = '{artifact_name}',
        num_concurrent_uploads: int = 4,
        upload_staging_folder: Optional[str] = None,
        use_procs: bool = True,
    ) -> None:
        self.provider = provider
        self.container = container
        self.provider_kwargs = provider_kwargs
        if should_log_artifact is None:
            should_log_artifact = _always_log
        self.should_log_artifact = should_log_artifact
        self.object_name_format = object_name_format
        self._run_name = None

        if upload_staging_folder is None:
            self._tempdir = tempfile.TemporaryDirectory()
            self._upload_staging_folder = self._tempdir.name
        else:
            self._tempdir = None
            self._upload_staging_folder = upload_staging_folder

        if num_concurrent_uploads < 1:
            raise ValueError("num_concurrent_uploads must be >= 1. Blocking uploads are not supported.")
        self._num_concurrent_uploads = num_concurrent_uploads

        if use_procs:
            mp_ctx = multiprocessing.get_context('spawn')
            self._file_upload_queue: Union[queue.Queue[Tuple[str, str, bool]],
                                           multiprocessing.JoinableQueue[Tuple[str, str,
                                                                               bool]]] = mp_ctx.JoinableQueue()
            self._finished_cls: Union[Callable[[], multiprocessing._EventType], Type[threading.Event]] = mp_ctx.Event
            self._proc_class = mp_ctx.Process
        else:
            self._file_upload_queue = queue.Queue()
            self._finished_cls = threading.Event
            self._proc_class = threading.Thread
        self._finished: Optional[Union[multiprocessing._EventType, threading.Event]] = None
        self._workers: List[Union[SpawnProcess, threading.Thread]] = []

    def init(self, state: State, logger: Logger) -> None:
        del state  # unused
        if self._finished is not None:
            raise RuntimeError("The ObjectStoreLogger is already initialized.")
        self._finished = self._finished_cls()
        self._run_name = logger.run_name
        object_name_to_test = self._format_object_name(".credentials_validated_successfully")
        _validate_credentials(self.provider, self.container, self.provider_kwargs, object_name_to_test)
        assert len(self._workers) == 0, "workers should be empty if self._finished was None"
        for _ in range(self._num_concurrent_uploads):
            worker = self._proc_class(
                target=_upload_worker,
                kwargs={
                    "file_queue": self._file_upload_queue,
                    "is_finished": self._finished,
                    "provider": self.provider,
                    "container": self.container,
                    "provider_kwargs": self.provider_kwargs,
                },
            )
            worker.start()
            self._workers.append(worker)

    def batch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._check_workers()

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._check_workers()

    def _check_workers(self):
        # Periodically check to see if any of the upload workers crashed
        # They would crash if:
        #   a) A file could not be uploaded, and the retry counter failed, or
        #   b) allow_overwrite=False, but the file already exists,
        for worker in self._workers:
            if not worker.is_alive():
                raise RuntimeError("Upload worker crashed. Please check the logs.")

    def log_file_artifact(self, state: State, log_level: LogLevel, artifact_name: str, file_path: pathlib.Path, *,
                          overwrite: bool):
        if not self.should_log_artifact(state, log_level, artifact_name):
            return
        copied_path = os.path.join(self._upload_staging_folder, str(uuid.uuid4()))
        copied_path_dirname = os.path.dirname(copied_path)
        os.makedirs(copied_path_dirname, exist_ok=True)
        shutil.copy2(file_path, copied_path)
        object_name = self._format_object_name(artifact_name)
        self._file_upload_queue.put_nowait((copied_path, object_name, overwrite))

    def post_close(self):
        # Cleaning up on post_close to ensure that all artifacts are uploaded
        if self._finished is not None:
            self._finished.set()
        for worker in self._workers:
            worker.join()
        if self._tempdir is not None:
            self._tempdir.cleanup()
        self._workers.clear()

    def get_uri_for_artifact(self, artifact_name: str) -> str:
        """Get the object store provider uri for an artfact.

        Args:
            artifact_name (str): The name of an artifact.

        Returns:
            str: The uri corresponding to the uploaded location of the artifact.
        """
        obj_name = self._format_object_name(artifact_name)
        provider_prefix = f"{self.provider}://{self.container}/"
        return provider_prefix + obj_name.lstrip("/")

    def _format_object_name(self, artifact_name: str):
        """Format the ``artifact_name`` according to the ``object_name_format_string``."""
        if self._run_name is None:
            raise RuntimeError("The run name is not set. It should have been set on Event.INIT.")
        key_name = self.object_name_format.format(
            rank=dist.get_global_rank(),
            local_rank=dist.get_local_rank(),
            world_size=dist.get_world_size(),
            local_world_size=dist.get_local_world_size(),
            node_rank=dist.get_node_rank(),
            artifact_name=artifact_name,
            run_name=self._run_name,
        )
        key_name = key_name.lstrip('/')

        return key_name


def _validate_credentials(
    provider: str,
    container: str,
    provider_kwargs: Optional[Dict[str, Any]],
    object_name_to_test: str,
) -> None:
    # Validates the credentails by attempting to touch a file in the bucket
    # raises a LibcloudError if there was a credentials failure.
    object_store = ObjectStore(provider=provider, container=container, provider_kwargs=provider_kwargs)
    object_store.upload_object_via_stream(
        obj=b"credentials_validated_successfully",
        object_name=object_name_to_test,
    )


def _upload_worker(
    file_queue: Union[queue.Queue[Tuple[str, str, bool]], multiprocessing.JoinableQueue[Tuple[str, str, bool]]],
    is_finished: Union[multiprocessing._EventType, threading.Event],
    provider: str,
    container: str,
    provider_kwargs: Optional[Dict[str, Any]],
):
    """A long-running function to handle uploading files to the object store specified by (``provider``, ``container``,
    ``provider_kwargs``).

    The worker will continuously poll ``file_queue`` for files to upload. Once ``is_finished`` is set, the worker will
    exit once ``file_queue`` is empty.
    """
    object_store = ObjectStore(provider=provider, container=container, provider_kwargs=provider_kwargs)
    while True:
        try:
            file_path_to_upload, object_name, overwrite = file_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            if is_finished.is_set():
                break
            else:
                continue
        if not overwrite:
            try:
                object_store.get_object_size(object_name)
            except ObjectDoesNotExistError:
                # Good! It shouldn't exist.
                pass
            else:
                # Exceptions will be detected on the next batch_end or epoch_end event
                raise FileExistsError(
                    textwrap.dedent(f"""\
                    {provider}://{container}/{object_name} already exists,
                    but allow_overwrite was set to False."""))
        log.info("Uploading file %s to %s://%s/%s", file_path_to_upload, object_store.provider_name,
                 object_store.container_name, object_name)
        retry_counter = 0
        while True:
            try:
                object_store.upload_object(
                    file_path=file_path_to_upload,
                    object_name=object_name,
                )
            except (LibcloudError, ProtocolError, TimeoutError, ConnectionError) as e:
                if isinstance(e, LibcloudError):
                    # The S3 driver does not encode the error code in an easy-to-parse manner
                    # So first checking if the error code is non-transient
                    is_transient_error = any(x in str(e) for x in ("408", "409", "425", "429", "500", "503", '504'))
                    if not is_transient_error:
                        raise e
                if retry_counter < 4:
                    retry_counter += 1
                    # exponential backoff
                    sleep_time = 2**(retry_counter - 1)
                    log.warning("Request failed. Sleeping %s seconds and retrying",
                                sleep_time,
                                exc_info=e,
                                stack_info=True)
                    time.sleep(sleep_time)
                    continue
                raise e

            os.remove(file_path_to_upload)
            file_queue.task_done()
            break
