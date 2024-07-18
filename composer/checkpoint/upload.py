# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Useful functions for uploading checkpoints to remote object store."""

import logging
import os
import dist
import tempfile
import pathlib
from typing import Optional
from concurrent.futures import Future

from composer.utils import (
    RemoteUploader,
    create_symlink_file,
)

log = logging.getLogger(__name__)

class CheckpointUploadAwaitable:
    def __init__(
        self,
        remote_uploader: RemoteUploader,
        remote_file_names: list[str],
        file_upload_future: Optional[Future] = None,
        symlink_file_name: Optional[str] = None,
        symlink_remote_file_name: Optional[str] = None
    ):
        self.remote_file_names = remote_file_names
        self.remote_uploader = remote_uploader
        self.file_upload_future = file_upload_futures
        self.symlink_file_name = symlink_file_name
        self.symlink_remote_file_name = symlink_remote_file_name

    def wait(self):
        log.debug(f'Waiting for uploading checkpoint file on current rank')
        if self.file_upload_future is not None:
            self.file_upload_future.result()
        log.debug(f'Current rank checkpoint uploading finishes')
        if self.symlink_file_name is None:
            return
        log.debug(f'Checking if all ranks finish uploading')
        remote_uploader.check_remote_files_exist_async(
            remote_checkpoint_file_names = self.remote_file_names
        ).result()
        log.debug(f'All ranks finish checkpoint uploading, start to upload symlink file')
        symlink_upload_future = remote_uploader.upload_file_async(
            remote_file_name=self.symlink_remote_file_name
            file_path=self.symlink_file_name,
            overwrite=True,
        ).result()
        log.debug(f'Symlink file uploading finishes!')


def upload_file(
    source_path: Optional[str]=None, 
    dest_dir: str,
    symlink_granularity: Optional[str]=None, # file, dir, or None
    symlink_name: Optional[str]='latest.symlink',
    async_upload: bool = True,
    state: optional[State] = None,
    overwrite: bool = False,
):
    """Standalone function for uploading a checkpoint file. 

    This function does not actually upload the checkpoint; it initiates the RemoteUploader's uploading of it
    Gets from state.callbacks or intializes RUD (and adds to callbacks) and calls upload_file

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
        state (optional[State]): If async_upload is True, then state must be specified so that the remote_uploader can be
            either extracted from state.callbacks or initialized and added to state.callbacks.
        overwrite (bool): If allow overwrite existing remote checkpoint files
"""
    
    remote_uploader = RemoteUploader(remote_folder=dest_dir)
    if source_path is not None:
        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
    else:
        dest_path = None
    all_dest_paths = dist.all_gather_object([dest_path] if dest_path is not None else [])
    all_dest_paths_list = []
    for dest_paths in all_dest_paths:
        all_dest_paths_list += dest_paths

    checkpoint_file_upload_future = None
    if dest_path is not None:
        assert remote_uploader is not None
        checkpoint_file_upload_future = remote_uploader.upload_file_async(
            remote_file_name=dest_path,
            file_path=dest_path,
            overwrite=overwrite,
        )

    
    symlink_remote_file_name = os.path.join(dest_dir, symlink_name)
    if dist.get_global_rank() == 0:     
        if symlink_granularity == 'file':
            create_symlink_file(dest_path, symlink_name)
        elif symlink_granularity == 'dir':
            create_symlink_file(str(pathlib.Path(dest_path).parent), symlink_name)
        elif symlink_granularity is not None:
            raise ValueError(f'Unrecognized symlink granularity: {symlink_granularity}')
    else:
        symlink_remote_file_name = None
        symlink_name = None

    upload_awaitable = CheckpointUploadAwaitable(
        remote_uploader=remote_uploader,
        remote_file_names=all_dest_paths_list,
        file_upload_future=checkpoint_file_upload_future,
        symlink_file_name=symlink_file_name,
        symlink_remote_file_name=symlink_remote_file_name,
    )

    if async_upload:
        return upload_awaitable
    else:
        upload_awaitable.wait()
