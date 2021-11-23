# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import multiprocessing
import os
import queue
import shutil
import tempfile
import threading
import time
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union

from composer.core.callback import RankZeroCallback
from composer.core.event import Event
from composer.core.logging import Logger
from composer.core.logging.logger import LogLevel
from composer.core.state import State
from composer.utils.run_directory import get_run_directory


class RunDirectoryUploader(RankZeroCallback):
    """Callback to upload the run directory to a blob store.

    This callback checks the run directory for new or modified files
    at the end of every epoch, and after every `upload_every_n_batches` batches.
    This callback detects new or modified files based off of the file modification
    timestamp. Only files that have a newer last modified timestamp since the last upload
    will be  uploaded.

    This uploader is compatible with multi-GPU training. It blocks the main thread
    for each local rank  when creating a copy of the modified files in the run directory
    before yielding back to the training loop. Uploads are performed from the copied files.
    It assumes that only the main thread on each rank writes to the run directory.

    While all uploads happen in the background, here are some additional tips for minimizing
    the performance impact:

        * Ensure that `upload_every_n_batches` is sufficiently infrequent as to limit when
          the blocking scans of the run direcory and copies of modified files.
          However, do not make it too infrequent in case if the training process unexpectedly dies,
          since data from the last upload may be lost.

        * Set `use_procs=True` (the default) to use background processes,
          instead of threads, to perform the file uploads. Processes are recommended to 
          ensure that the GIL is not blocking the training loop when performance CPU
          operations on uploaded files (e.g. comparing and computing checksums).
          Network I/O happens always occurs in the background.

        * Provide a RAM disk path for the `upload_staging_folder` parameter. Copying files to stage on RAM
          will be faster than writing to disk. However, you must have sufficient excess RAM on your system,
          or you may experience OutOfMemory errors.

    .. note::

        To use this callback, install composer with `pip install mosaicml[logging]`.

    Args:
        provider (str): Cloud provider to use.

            Specify the last part of the Apache Libcloud Module here.
            `This document <https://libcloud.readthedocs.io/en/stable/storage/supported_providers.html#provider-matrix>`
            lists all supported providers. For example, the module name for Amazon S3 is `libcloud.storage.drivers.s3`, so
            to use S3, specify 's3' here.

        container (str): The name of the container (i.e. bucket) to use.
        num_concurrent_uploads (int, optional): Maximum number of concurrent uploads. Defaults to 4.
        upload_staging_folder (Optional[str], optional): A folder to use for staging uploads.
            If not specified, defaults to using a :class:`~tempfile.TemporaryDirectory`.
        use_procs (bool, optional): Whether to perform file uploads in background processes (as opposed to threads).
            Defaults to True.
        upload_every_n_batches (int, optional): Interval at which to scan the run directory for changes and to
            queue uploads of files. Uploads are always queued at the end of the epoch. Defaults to every 100 batches.
        provider_init_kwargs (Dict[str, Any], optional): Parameters to pass into the constructor for the
            :class:`~libcloud.storage.providers.Provider` constructor. These arguments would usually include the cloud region
            and credentials. Defaults to None, which is equivalent to an empty dictionary.
    """

    def __init__(
        self,
        provider: str,
        container: Optional[str] = None,
        num_concurrent_uploads: int = 4,
        upload_staging_folder: Optional[str] = None,
        use_procs: bool = True,
        upload_every_n_batches: int = 100,
        provider_init_kwargs: Dict[str, Any] = None,
    ) -> None:
        run_directory = get_run_directory()
        if run_directory is None:
            warnings.warn("NoRunDirectory: The run directory is not set, so the RunDirectoryUploader will be a no-op")
            return

        if provider_init_kwargs is None:
            provider_init_kwargs = {}
        self._provider_init_kwargs = provider_init_kwargs
        self._upload_every_n_batches = upload_every_n_batches
        self._object_name_prefix = ""  # TODO ravi. Decide how this will be set. Hparams? Run directory name?

        self._last_upload_timestamp = 0.0  # unix timestamp of last uploaded time
        if upload_staging_folder is None:
            self._tempdir = tempfile.TemporaryDirectory()
            self._upload_staging_folder = self._tempdir.name
        else:
            self._tempdir = None
            self._upload_staging_folder = upload_staging_folder

        if num_concurrent_uploads < 1:
            raise ValueError("num_concurrent_uploads must be >= 1. Blocking uploads are not supported.")
        self._num_concurrent_uploads = num_concurrent_uploads
        self._provider = provider
        self._container = container

        if use_procs:
            self._file_upload_queue: Union[queue.Queue[str],
                                           multiprocessing.JoinableQueue[str]] = multiprocessing.JoinableQueue()
            self._finished_cls: Union[Callable[[], multiprocessing._EventType],
                                      Type[threading.Event]] = multiprocessing.Event
            self._proc_class = multiprocessing.Process
        else:
            self._file_upload_queue = queue.Queue()
            self._finished_cls = threading.Event
            self._proc_class = threading.Thread
        self._finished: Union[None, multiprocessing._EventType, threading.Event] = None
        self._workers = []

    def _init(self) -> None:
        self._finished = self._finished_cls()
        self._last_upload_timestamp = 0.0
        self._workers = [
            self._proc_class(target=_upload_worker,
                             kwargs={
                                 "file_queue": self._file_upload_queue,
                                 "is_finished": self._finished,
                                 "upload_staging_dir": self._upload_staging_folder,
                                 "provider_name": self._provider,
                                 "container_name": self._container,
                                 "object_name_prefix": self._object_name_prefix,
                                 "init_kwargs": self._provider_init_kwargs,
                             }) for _ in range(self._num_concurrent_uploads)
        ]
        for worker in self._workers:
            worker.start()

    def _run_event(self, event: Event, state: State, logger: Logger) -> None:
        if get_run_directory() is None:
            return
        if event == Event.INIT:
            self._init()
        if event == Event.BATCH_END:
            if (state.batch_idx + 1) % self._upload_every_n_batches == 0:
                self._trigger_upload(state, logger, LogLevel.BATCH)
        if event == Event.EPOCH_END:
            self._trigger_upload(state, logger, LogLevel.EPOCH)
        if event == Event.TRAINING_END:
            self._trigger_upload(state, logger, LogLevel.FIT)
            # TODO -- we are missing logfiles from other callbacks / loggers that write on training end but after
            # the run directory uploader is invoked. This callback either needs to fire last,
            # or we need another event such as cleanup
            self._close()

    def _close(self):
        assert self._finished is not None, "finished should not be None"
        self._finished.set()
        for worker in self._workers:
            worker.join()

    def _trigger_upload(self, state: State, logger: Logger, log_level: LogLevel) -> None:
        # Ensure that every rank is at this point
        # Assuming only the main thread on each rank writes to the run directory, then the barrier here will ensure
        # that the run directory is not being modified after we pass this barrier
        # TODO(ravi) -- add in a ddp barrier here.
        # state.ddp.barrier()
        new_last_uploaded_timestamp = time.time()
        # Now, for each file that was modified since self._last_upload_timestamp, copy it to the temporary directory
        # IMPROTANT: From now, until self._last_upload_timestamp is updated, no files should be written to the run directory
        run_directory = get_run_directory()
        assert run_directory is not None, "invariant error"
        files_to_be_uploaded = []
        for root, dirs, files in os.walk(run_directory):
            del dirs  # unused
            for file in files:
                filepath = os.path.join(root, file)
                relpath = os.path.relpath(filepath, run_directory)  # chop off the run directory
                modified_time = os.path.getmtime(filepath)
                if modified_time > self._last_upload_timestamp:
                    copied_path = os.path.join(self._upload_staging_folder, str(new_last_uploaded_timestamp), relpath)
                    files_to_be_uploaded.append(relpath)
                    copied_path_dirname = os.path.dirname(copied_path)
                    os.makedirs(copied_path_dirname, exist_ok=True)
                    # shutil.copyfile(filepath, copied_path)
                    shutil.copy2(filepath, copied_path)
                    self._file_upload_queue.put_nowait(copied_path)
        self._last_upload_timestamp = new_last_uploaded_timestamp
        # now log which files are being uploaded. OK to do, since we're done reading the directory,
        # and any logfiles will now have their last modified timestamp
        # incremented past self._last_upload_timestamp
        logger.metric(log_level, {"run_directory/uploaded_files": files_to_be_uploaded})


def _upload_worker(
    file_queue: Union[queue.Queue[str], multiprocessing.JoinableQueue[str]],
    is_finished: Union[multiprocessing._EventType, threading.Event],
    upload_staging_dir: str,
    provider_name: str,
    container_name: str,
    object_name_prefix: str,
    init_kwargs: Dict[str, Any],
):
    """A long-running function to handle uploading files.

    Args:
        file_queue (queue.Queue or multiprocessing.JoinableQueue): The worker will poll
            this queue for files to upload.
        is_finished (threading.Event or multiprocessing.Event): An event that will be
            set when training is finished and no new files will be added to the queue.
            The worker will continue to upload existing files that are in the queue.
            When the queue is empty, the worker will exit.
        upload_staging_dir (str): The upload staging directory.
        provider_name (str): The cloud provider name.
        container_name (str): The container name (e.g. s3 bucket) for the provider
            where files will be uploaded.
        object_name_prefix (str): Prefix to prepend to the object names
             before they are uploaded to the blob store.
        init_kwargs (Dict[str, Any]): Arguments to pass in to the
            :class:`~libcloud.storage.providers.Provider` constructor.
    """
    from libcloud.storage.providers import get_driver
    provider_cls = get_driver(provider_name)
    provider = provider_cls(**init_kwargs)
    container = provider.get_container(container_name)
    while True:
        try:
            file_path_to_upload = file_queue.get_nowait()
        except queue.Empty:
            if is_finished.is_set():
                break
            else:
                time.sleep(0.5)
                continue
        obj_name = ",".join(os.path.relpath(file_path_to_upload, upload_staging_dir).split(
            os.path.sep)[1:])  # the first folder is the upload timestamp. Chop that off.
        provider.upload_object(
            file_path=file_path_to_upload,
            container=container,
            object_name=object_name_prefix + obj_name,
        )
        os.remove(file_path_to_upload)
        file_queue.task_done()
