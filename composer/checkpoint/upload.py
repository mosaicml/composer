# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for uploading checkpoints to remote object store."""

import logging
import os
import pathlib
import time
from concurrent.futures import Future
from typing import Optional

import composer.utils.dist as dist
from composer.core import Callback, State
from composer.loggers import Logger, MosaicMLLogger
from composer.utils import (
    RemoteFilesExistingCheckStatus,
    RemoteUploader,
    create_symlink_file,
    parse_uri,
)

log = logging.getLogger(__name__)


class CheckpointUploadCallback(Callback):
    """Callback class for async checkpoint uploading."""

    def __init__(
        self,
        remote_uploader: RemoteUploader,
        upload_timeout_in_seconds: int = 3600,
    ):
        self.remote_uploader = remote_uploader
        if dist.get_global_rank() == 0:
            self.rank_saves_symlinks = True
        else:
            self.rank_saves_symlinks = False

        # Each Tuple contains information of:
        # the future of checking if checkpoint files upload finish
        # local symlink file name
        # remote symlink file name
        self.symlink_upload_tasks: list[tuple[Future[RemoteFilesExistingCheckStatus], str, Optional[str]]] = []
        self.upload_timeout_in_seconds: int = upload_timeout_in_seconds

        # Allow unit test to override this to make it faster
        self._symlink_upload_wait_before_next_try_in_seconds = 30.0

    def add_symlink_upload_task(
        self,
        remote_file_names: list[str],
        symlink_file_name: Optional[str] = None,
        symlink_remote_file_name: Optional[str] = None,
    ):
        check_remote_files_exist_future = None
        if self.rank_saves_symlinks and symlink_file_name is not None:
            check_remote_files_exist_future = self.remote_uploader.check_remote_files_exist_async(
                remote_checkpoint_file_names=remote_file_names,
                max_wait_time_in_seconds=self.upload_timeout_in_seconds,
                wait_before_next_try_in_seconds=self._symlink_upload_wait_before_next_try_in_seconds,
            )
            self.symlink_upload_tasks.append(
                (check_remote_files_exist_future, symlink_file_name, symlink_remote_file_name),
            )

    def _log_checkpoint_upload(self, logger: Logger):
        for destination in logger.destinations:
            if isinstance(destination, MosaicMLLogger):
                destination.log_metadata({'checkpoint_uploaded_time': time.time()}, force_flush=True)

    def batch_end(self, state: State, logger: Logger) -> None:
        del state  # unused
        self.remote_uploader.check_workers()
        if not self.rank_saves_symlinks:
            return
        undone_symlink_upload_tasks = []
        for (check_remote_files_exist_future, local_symlink_file,
             remote_symlink_file) in reversed(self.symlink_upload_tasks):
            if not check_remote_files_exist_future.done():
                undone_symlink_upload_tasks.insert(
                    0,
                    (check_remote_files_exist_future, local_symlink_file, remote_symlink_file),
                )
                continue
            else:
                result = check_remote_files_exist_future.result()
                if result == RemoteFilesExistingCheckStatus.EXIST:
                    assert remote_symlink_file is not None
                    self.remote_uploader.upload_file_async(
                        remote_file_name=remote_symlink_file,
                        file_path=pathlib.Path(local_symlink_file),
                        overwrite=True,
                    )
                    self._log_checkpoint_upload(logger)
                    break
                else:
                    raise RuntimeError(f'Failed to check if checkpoint files upload finish: {result}')
        self.symlink_upload_tasks = undone_symlink_upload_tasks

    def wait(self) -> None:
        log.info('Waiting for checkpoint uploading to finish')
        self.remote_uploader.wait()
        if self.rank_saves_symlinks and len(self.symlink_upload_tasks) > 0:
            log.debug('Uploading symlink to the latest checkpoint')
            # We only need to upload a symlink pointing to the latest checkpoint files, so we can ignore successful uploads of older checkpoints.
            check_remote_files_exist_future, local_symlink_file, remote_symlink_file = self.symlink_upload_tasks[-1]
            result = check_remote_files_exist_future.result()
            if result == RemoteFilesExistingCheckStatus.EXIST:
                assert remote_symlink_file is not None
                symlink_upload_future = self.remote_uploader.upload_file_async(
                    remote_file_name=remote_symlink_file,
                    file_path=pathlib.Path(local_symlink_file),
                    overwrite=True,
                )
                symlink_upload_future.result()
            else:
                raise RuntimeError(f'Failed to check if checkpoint files upload finish: {result}')
            self.symlink_upload_tasks = []
        log.info('Checkpoint uploading finished!')

    def fit_end(self, state: State, logger: Logger) -> None:
        del state  # unused
        self.wait()
        self._log_checkpoint_upload(logger)

    def post_close(self):
        # Wait the symlink file upload to finish and close remote uploader
        try:
            self.remote_uploader.wait_and_close()
        except Exception as e:
            log.error(f'RemoteUploader run into exception {e}')



def upload_file(
    dest_dir: str,
    source_path: Optional[str]=None,
    symlink_granularity: Optional[str]=None, # file, dir, or None
    symlink_name: Optional[str]='latest.symlink',
    async_upload: bool = True,
    state: Optional[State] = None,
    overwrite: bool = False,
):
    """Standalone function for uploading a checkpoint file.

    This function does not actually upload the checkpoint; it initiates the RemoteUploader's uploading of it
    Args:
        source_path (str): The path to the file to upload.
        dest_dir (str): The directory/uri to upload the file to.
        symlink_granularity (Optional[str]): The granularity to use for symlinking. One of 'file', 'dir', or None.
            if None: no symlink uploaded
            if 'file': command remoteuploader to wait until the file (specificied by source_path) is uploaded and then uploads a symlink pointing to the uploaded file
            if 'dir': command remoteuploader  to wait until all files across all ranks are uploaded to dest_dir and then uploads a symlink
                pointing to the remote directory (prefix in object_store terminology).
        symlink_name (Optional[str]): The name to use for the symlink. Defaults to 'latest.symlink'.
        async_upload (bool): If True, the uploads will be done asynchronously via the RemoteUploader and this function will return immediately.
        state (Optional[State]): If async_upload is True, then state must be specified so that the remote_uploader can be
            either extracted from state.callbacks or initialized and added to state.callbacks.
        overwrite (bool): If allow overwrite existing remote checkpoint files
    """
    remote_uploader = RemoteUploader(remote_folder=dest_dir)
    dest_path = ''
    if source_path is not None:
        _, _, dest_path = parse_uri(dest_dir)
        remote_file_name = os.path.join(dest_path, os.path.basename(source_path))
    else:
        remote_file_name = None
    all_remote_file_names = dist.all_gather_object([remote_file_name] if remote_file_name is not None else [])
    remote_file_names = []
    for filenames in all_remote_file_names:
        remote_file_names += filenames

    if source_path is not None:
        assert remote_uploader is not None
        assert remote_file_name is not None
        remote_uploader.upload_file_async(
            remote_file_name=remote_file_name,
            file_path=pathlib.Path(source_path),
            overwrite=overwrite,
        )
    symlink_remote_file_name = None
    if dist.get_global_rank() == 0:
        if symlink_name is not None:
            symlink_remote_file_name = os.path.join(dest_path, symlink_name)
            if symlink_granularity == 'file':
                create_symlink_file(dest_path, symlink_name)
            elif symlink_granularity == 'dir':
                create_symlink_file(str(pathlib.Path(dest_path).parent), symlink_name)
            elif symlink_granularity is not None:
                raise ValueError(f'Unrecognized symlink granularity: {symlink_granularity}')
    else:
        symlink_remote_file_name = None
        symlink_name = None

    upload_callback = None
    if state is not None:
        for callback in state.callbacks:
            if isinstance(callback, CheckpointUploadCallback):
                upload_callback = callback
                break
    if upload_callback is None:
        upload_callback = CheckpointUploadCallback(remote_uploader=remote_uploader)
        if state is not None:
            state.callbacks.append(upload_callback)
    upload_callback.add_symlink_upload_task(
        remote_file_names=remote_file_names,
        symlink_file_name=symlink_name,
        symlink_remote_file_name=symlink_remote_file_name,
    )

    if not async_upload:
        upload_callback.wait()
