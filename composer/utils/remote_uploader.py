# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Util class to upload file."""

import logging
import multiprocessing
import os
import pathlib
import queue
import shutil
import tempfile
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from typing import List

from composer.utils.dist import get_local_rank
from composer.utils.file_helpers import (
    maybe_create_object_store_from_uri,
)
from composer.utils.object_store.object_store import ObjectStoreTransientError
from composer.utils.retrying import retry

log = logging.getLogger(__name__)

__all__ = ['RemoteUploader']


def _upload_worker(
    remote_folder: str,
    file_queue: queue.Queue,
    is_closed_event: multiprocessing.Event,  # pyright: ignore[reportGeneralTypeIssues]
    num_attempts: int,
    parent_process_id: int,
) -> None:
    object_store = maybe_create_object_store_from_uri(remote_folder)
    if object_store is None:
        raise RuntimeError(f'Can not create object store from remote folder {remote_folder}')

    @retry(ObjectStoreTransientError, num_attempts=num_attempts)
    def upload_file(
        remote_file_name: str,
        local_file_path: str,
        overwrite: bool,
        retry_index: int = 0,
    ):
        if retry_index == 0 and not overwrite:
            try:
                object_store.get_object_size(remote_file_name)
            except FileNotFoundError:
                # Good! It shouldn't exist.
                pass
            else:
                raise FileExistsError(f'Object {remote_file_name} already exists, but overwrite was set to False.')
        log.info(f'Uploading file {local_file_path} to {remote_file_name}')
        object_store.upload_object(
            object_name=remote_file_name,
            filename=local_file_path,
        )
        os.remove(local_file_path)

    def is_parent_process_dead():
        try:
            # Sending signal 0 to ppid will raise OSError if ppid is not running
            # and do nothing otherwise
            os.kill(parent_process_id, 0)
        except OSError:
            return True
        return False

    while True:
        try:
            file_path_to_upload, remote_file_name, overwrite = file_queue.get(block=True, timeout=0.5)
        except queue.Empty:
            if is_closed_event.is_set():
                break
            elif is_parent_process_dead():
                log.warning(f'Terminate the uploader worker since parent process died')
                break
            else:
                continue

        # When encountering issues with too much concurrency in uploads, staggering the uploads can help.
        # This stagger is intended for use when uploading model shards from every rank, and will effectively reduce
        # the concurrency by a factor of num GPUs per node.
        local_rank = get_local_rank()
        local_rank_stagger = int(os.environ.get('COMPOSER_LOCAL_RANK_STAGGER_SECONDS', 0))
        log.debug(f'Staggering uploads by {local_rank * local_rank_stagger} seconds on {local_rank} local rank.')
        time.sleep(local_rank * local_rank_stagger)

        upload_file(remote_file_name, file_path_to_upload, overwrite)


class RemoteUploader:
    """Class for uploading a file to object store asynchronously."""

    def __init__(
        self,
        remote_folder: str,
        num_concurrent_uploads: int = 1,
        num_attempts: int = 3,
    ):
        assert num_concurrent_uploads >= 1, f'num_concurrent_uploads must be >= 1, got {num_concurrent_uploads}'

        # A folder to use for staging uploads
        self._tempdir = tempfile.TemporaryDirectory()
        self._upload_staging_folder = self._tempdir.name

        self.num_attempts = num_attempts

        mp_ctx = multiprocessing.get_context('spawn')
        manager = mp_ctx.Manager()
        self.file_queue: queue.Queue = manager.Queue()
        self.executor = ProcessPoolExecutor(
            max_workers=num_concurrent_uploads,
            mp_context=mp_ctx,
        )
        self.is_closed_event = manager.Event()
        self.workers: List[Future] = []
        for _ in range(num_concurrent_uploads):
            self.workers.append(
                self.executor.submit(
                    _upload_worker,
                    remote_folder=remote_folder,
                    file_queue=self.file_queue,
                    is_closed_event=self.is_closed_event,
                    num_attempts=num_attempts,
                    parent_process_id=os.getpid(),
                ),
            )

    def upload_file_async(
        self,
        remote_file_name: str,
        file_path: pathlib.Path,
        overwrite: bool,
    ):
        """Async call to submit a job for uploading."""
        # Copy file to staging folder
        copied_path = os.path.join(self._upload_staging_folder, str(uuid.uuid4()))
        os.makedirs(self._upload_staging_folder, exist_ok=True)
        shutil.copy2(file_path, copied_path)

        self.file_queue.put_nowait((copied_path, remote_file_name, overwrite))

    def check_workers(self):
        """Check if any upload_worker has exception.

        Traverse self.workders, raise the worker exception to main process
        """
        for future in self.workers:
            if future.done():
                # future.exception is a blocking call
                exception_or_none = future.exception()
                if exception_or_none is not None:
                    raise exception_or_none

    def wait_and_close(self):
        """Blocking call to wait all uploading to finish and close this uploader.

        After this function is called, users can not use this uploader
        to uploading anymore. So please only call wait_and_close() after submitting
        all uploading requests.
        """
        # Send signal to upload_workers to stop
        self.is_closed_event.set()
        # Blocking call to wait the upload_worker to finish
        for future in self.workers:
            future.result()
        self.executor.shutdown()
