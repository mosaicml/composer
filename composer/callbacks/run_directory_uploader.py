# Copyright 2021 MosaicML. All Rights Reserved.

"""Periodically upload :mod:`~composer.utils.run_directory` to a blob store during training."""
from __future__ import annotations

import datetime
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
from typing import Callable, Optional, Tuple, Type, Union

from libcloud.common.types import LibcloudError
from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError

from composer.core.callback import Callback
from composer.core.logging import Logger
from composer.core.logging.logger import LogLevel
from composer.core.state import State
from composer.utils import dist, run_directory
from composer.utils.object_store import ObjectStoreProviderHparams

log = logging.getLogger(__name__)

__all__ = ["RunDirectoryUploader"]


class RunDirectoryUploader(Callback):
    """Callback to upload the run directory to a blob store.

    This callback checks the run directory for new or modified files at the end of every epoch, and after every
    ``upload_every_n_batches`` batches.  This callback detects new or modified files based on the file modification
    timestamp. Only files that have a newer last modified timestamp since the last upload will be  uploaded.

    Example
        .. testsetup:: composer.callbacks.RunDirectoryUploader.__init__

           import os
           import functools
           from composer.callbacks import RunDirectoryUploader, run_directory_uploader

           # For this example, we do not validate credentials
           def do_not_validate(
               object_store_provider_hparams: ObjectStoreProviderHparams,
               object_name_prefix: str,
           ) -> None:
               pass

           run_directory_uploader._validate_credentials = do_not_validate
           
           os.environ['OBJECT_STORE_KEY'] = 'KEY'
           os.environ['OBJECT_STORE_SECRET'] = 'SECRET'
           RunDirectoryUploader = functools.partial(
               RunDirectoryUploader,
               use_procs=False,
               num_concurrent_uploads=1,
           )

        .. doctest:: composer.callbacks.RunDirectoryUploader.__init__

           >>> osphparams = ObjectStoreProviderHparams(
           ...     provider="s3",
           ...     container="run-dir-test",
           ...     key_environ="OBJECT_STORE_KEY",
           ...     secret_environ="OBJECT_STORE_SECRET",
           ...     region="us-west-2",
           ...     )
           >>> # construct trainer object with this callback
           >>> run_directory_uploader = RunDirectoryUploader(osphparams)
           >>> trainer = Trainer(
           ...     model=model,
           ...     train_dataloader=train_dataloader,
           ...     eval_dataloader=eval_dataloader,
           ...     optimizers=optimizer,
           ...     max_duration="1ep",
           ...     callbacks=[run_directory_uploader],
           ... )
           >>> # trainer will run this callback whenever the EPOCH_END
           >>> # is triggered, like this:
           >>> _ = trainer.engine.run_event(Event.EPOCH_END)
        
        .. testcleanup:: composer.callbacks.RunDirectoryUploader.__init__

           # Shut down the uploader
           run_directory_uploader._finished.set()

    .. note::
        This callback blocks the training loop to copy files from the :mod:`~composer.utils.run_directory` to the
        ``upload_staging_folder`` and to queue these files to the upload queues of the workers. Actual upload happens in
        the background.  While all uploads happen in the background, here are some additional tips for minimizing the
        performance impact:

        * Ensure that ``upload_every_n_batches`` is sufficiently infrequent as to limit when the blocking scans of the
          run directory and copies of modified files.  However, do not make it too infrequent in case if the training
          process unexpectedly dies, since data written after the last upload may be lost.

        * Set ``use_procs=True`` (the default) to use background processes, instead of threads, to perform the file
          uploads. Processes are recommended to ensure that the GIL is not blocking the training loop when performance CPU
          operations on uploaded files (e.g. computing and comparing checksums).  Network I/O happens always occurs in the
          background.

        * Provide a RAM disk path for the ``upload_staging_folder`` parameter. Copying files to stage on RAM will be
          faster than writing to disk. However, you must have sufficient excess RAM on your system, or you may experience
          OutOfMemory errors.

    Args:
        object_store_provider_hparams (ObjectStoreProviderHparams): ObjectStoreProvider hyperparameters object

            See :class:`~composer.utils.object_store.ObjectStoreProviderHparams` for documentation.

        object_name_prefix (str, optional): A prefix to prepend to all object keys. An object's key is this prefix combined
            with its path relative to the run directory. If the container prefix is non-empty, a trailing slash ('/') will
            be added if necessary. If not specified, then the prefix defaults to the run directory. To disable prefixing,
            set to the empty string.

            For example, if ``object_name_prefix = 'foo'`` and there is a file in the run directory named ``bar``, then that file
            would be uploaded to ``foo/bar`` in the container.
        num_concurrent_uploads (int, optional): Maximum number of concurrent uploads. Defaults to 4.
        upload_staging_folder (str, optional): A folder to use for staging uploads.
            If not specified, defaults to using a :func:`~tempfile.TemporaryDirectory`.
        use_procs (bool, optional): Whether to perform file uploads in background processes (as opposed to threads).
            Defaults to True.
        upload_every_n_batches (int, optional): Interval at which to scan the run directory for changes and to
            queue uploads of files. In addition, uploads are always queued at the end of the epoch. Defaults to every 100 batches.
    """

    def __init__(
        self,
        object_store_provider_hparams: ObjectStoreProviderHparams,
        object_name_prefix: Optional[str] = None,
        num_concurrent_uploads: int = 4,
        upload_staging_folder: Optional[str] = None,
        use_procs: bool = True,
        upload_every_n_batches: int = 100,
    ) -> None:
        self._object_store_provider_hparams = object_store_provider_hparams
        self._upload_every_n_batches = upload_every_n_batches
        # get the name of the run directory, without the rank
        run_directory_name = os.path.basename(run_directory.get_node_run_directory())
        if object_name_prefix is None:
            self._object_name_prefix = f"{run_directory_name}/"
        else:
            if object_name_prefix == "":
                self._object_name_prefix = ""
            else:
                if not object_name_prefix.endswith('/'):
                    object_name_prefix = f"{object_name_prefix}/"
                self._object_name_prefix = object_name_prefix
        # Keep the subfoldering by rank
        self._object_name_prefix += f"rank_{dist.get_global_rank()}/"
        self._last_upload_timestamp = datetime.datetime.fromtimestamp(0)  # unix timestamp of last uploaded time
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
            self._file_upload_queue: Union[queue.Queue[Tuple[str, str]],
                                           multiprocessing.JoinableQueue[Tuple[str, str]]] = mp_ctx.JoinableQueue()
            self._finished_cls: Union[Callable[[], multiprocessing._EventType], Type[threading.Event]] = mp_ctx.Event
            self._proc_class = mp_ctx.Process
        else:
            self._file_upload_queue = queue.Queue()
            self._finished_cls = threading.Event
            self._proc_class = threading.Thread
        self._finished: Union[None, multiprocessing._EventType, threading.Event] = None
        self._workers = []

        _validate_credentials(object_store_provider_hparams, self._object_name_prefix)

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused
        self._finished = self._finished_cls()
        self._last_upload_timestamp = run_directory.get_run_directory_timestamp()
        self._workers = [
            self._proc_class(target=_upload_worker,
                             kwargs={
                                 "file_queue": self._file_upload_queue,
                                 "is_finished": self._finished,
                                 "object_store_provider_hparams": self._object_store_provider_hparams,
                                 "object_name_prefix": self._object_name_prefix,
                             }) for _ in range(self._num_concurrent_uploads)
        ]
        for worker in self._workers:
            worker.start()

    def batch_end(self, state: State, logger: Logger) -> None:
        if int(state.timer.batch_in_epoch) % self._upload_every_n_batches == 0:
            self._trigger_upload(logger, LogLevel.BATCH)

    def epoch_end(self, state: State, logger: Logger) -> None:
        del state  # unused
        self._trigger_upload(logger, LogLevel.EPOCH)

    def post_close(self):
        # Cleaning up on post_close to ensure that all artifacts are uploaded
        self._trigger_upload(logger=None, log_level=None)
        if self._finished is not None:
            self._finished.set()
        for worker in self._workers:
            worker.join()
        if self._tempdir is not None:
            self._tempdir.cleanup()

    def _trigger_upload(self, logger: Optional[Logger], log_level: Optional[LogLevel]) -> None:
        new_last_uploaded_timestamp = run_directory.get_run_directory_timestamp()

        # Now, for each file that was modified since self._last_upload_timestamp, copy it to the temporary directory
        files_to_be_uploaded = []

        # check if any upload threads have crashed. if so, then shutdown the training process
        for worker in self._workers:
            if not worker.is_alive():
                assert self._finished is not None, "invariant error"
                self._finished.set()
                raise RuntimeError("Upload worker crashed unexpectedly")
        modified_files = run_directory.get_modified_files(self._last_upload_timestamp)
        for filepath in modified_files:
            copied_path = os.path.join(self._upload_staging_folder, str(uuid.uuid4()))
            file_key_name = os.path.relpath(filepath, run_directory.get_run_directory())
            files_to_be_uploaded.append(file_key_name)
            copied_path_dirname = os.path.dirname(copied_path)
            os.makedirs(copied_path_dirname, exist_ok=True)
            shutil.copy2(filepath, copied_path)
            self._file_upload_queue.put_nowait((copied_path, file_key_name))

        self._last_upload_timestamp = new_last_uploaded_timestamp
        if logger is not None and log_level is not None:
            # now log which files are being uploaded. OK to do, since we're done reading the directory,
            # and any logfiles will now have their last modified timestamp
            # incremented past self._last_upload_timestamp
            logger.metric(log_level, {"run_directory/uploaded_files": files_to_be_uploaded})

    def get_uri_for_uploaded_file(self, local_filepath: Union[pathlib.Path, str]) -> str:
        """Get the object store provider uri for a specific local filepath.

        Args:
            local_filepath (Union[pathlib.Path, str]): The local file for which to get the uploaded uri.

        Returns:
            str: The uri corresponding to the upload location of the file.
        """
        rel_to_run_dir = os.path.relpath(local_filepath, run_directory.get_run_directory())
        obj_name = self._object_name_prefix + rel_to_run_dir
        provider_name = self._object_store_provider_hparams.provider
        container = self._object_store_provider_hparams.container
        provider_prefix = f"{provider_name}://{container}/"
        return provider_prefix + obj_name.lstrip("/")


def _validate_credentials(
    object_store_provider_hparams: ObjectStoreProviderHparams,
    object_name_prefix: str,
) -> None:
    # Validates the credentails by attempting to touch a file in the bucket
    provider = object_store_provider_hparams.initialize_object()
    provider.upload_object_via_stream(
        obj=b"credentials_validated_successfully",
        object_name=f"{object_name_prefix}.credentials_validated_successfully",
    )


def _upload_worker(
    file_queue: Union[queue.Queue[str], multiprocessing.JoinableQueue[str]],
    is_finished: Union[multiprocessing._EventType, threading.Event],
    object_store_provider_hparams: ObjectStoreProviderHparams,
    object_name_prefix: str,
):
    """A long-running function to handle uploading files.

    Args:
        file_queue (queue.Queue or multiprocessing.JoinableQueue): The worker will poll
            this queue for files to upload.
        is_finished (threading.Event or multiprocessing.Event): An event that will be
            set when training is finished and no new files will be added to the queue.
            The worker will continue to upload existing files that are in the queue.
            When the queue is empty, the worker will exit.
        object_store_provider_hparams (ObjectStoreProviderHparams): The configuration
            for the underlying object store provider.
        object_name_prefix (str): Prefix to prepend to the object names
             before they are uploaded to the blob store.
    """
    provider = object_store_provider_hparams.initialize_object()
    while True:
        try:
            file_path_to_upload, object_store_name = file_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            if is_finished.is_set():
                break
            else:
                continue
        obj_name = object_name_prefix + object_store_name
        log.info("Uploading file %s to %s://%s/%s", file_path_to_upload, provider.provider_name,
                 provider.container_name, obj_name)
        retry_counter = 0
        while True:
            try:
                provider.upload_object(
                    file_path=file_path_to_upload,
                    object_name=obj_name,
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
                    log.warn("Request failed. Sleeping %s seconds and retrying",
                             sleep_time,
                             exc_info=e,
                             stack_info=True)
                    time.sleep(sleep_time)
                    continue
                raise e

            os.remove(file_path_to_upload)
            file_queue.task_done()
            break
