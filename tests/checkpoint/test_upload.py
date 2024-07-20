# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import patch
import multiprocessing

import pytest
import torch

from composer.checkpoint import upload_file 
from composer.utils import dist
from tests.checkpoint.helpers import init_model
from tests.common.markers import world_size
from tests.utils.test_remote_uploader import DummyObjectStore
from unittest.mock import patch


@world_size(1, 2)
@pytest.mark.gpu
def test_upload_file(world_size: int):
    # Prepare local file to upload
    local_file_path = tempfile.TemporaryDirectory()
    rank = dist.get_global_rank() if world_size > 1 else 0
    file_name = f'checkpoint_file_rank_{rank}'
    local_file_full_name = os.path.join(local_file_path, file_name)
    with open(local_file_full_name, 'w') as f:
        f.write(str(rank))

    # Prepare remote path
    remote_path = tempfile.TemporaryDirectory()
    if world_size > 1:
        if rank == 0:
            remote_path_list = [remote_path]
        else:
            remote_path_list = []
        remote_path_list = dist.broadcast_object_list(remote_path_list)
        remote_path = remote_path_list[0]
    def _get_tmp_dir():
        return remote_path
    
    fork_context = multiprocessing.get_context('fork')
    with patch('composer.utils.object_store.utils.S3ObjectStore', DummyObjectStore):
        with patch('tempfile.TemporaryDirectory', _get_tmp_dir):
            with patch('composer.utils.remote_uploader.multiprocessing.get_context', lambda _: fork_context):
                upload_file(
                    source_path=local_file_full_name,
                    dest_dir='S3://bucket_name/',
                    symlink_granularity='file',
                )


