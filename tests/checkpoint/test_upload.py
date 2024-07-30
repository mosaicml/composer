# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import os
import tempfile
from unittest.mock import patch

import pytest

from composer.checkpoint import upload_file
from composer.core import Engine, Event, State
from composer.loggers import Logger
from composer.utils import dist
from tests.common.markers import world_size
from tests.utils.test_remote_uploader import DummyObjectStore


@world_size(1, 2)
@pytest.mark.gpu
@pytest.mark.parametrize(('async_upload'), [True, False])
def test_upload_file(world_size: int, async_upload, dummy_state: State):
    # Prepare local file to upload
    local_file_path = tempfile.TemporaryDirectory()
    rank = dist.get_global_rank() if world_size > 1 else 0
    file_name = f'checkpoint_file_rank_{rank}'
    local_file_full_name = os.path.join(local_file_path.name, file_name)
    with open(local_file_full_name, 'w') as f:
        f.write(str(rank))

    # Prepare remote path
    remote_path = tempfile.TemporaryDirectory()
    if world_size > 1:
        remote_path_list = [remote_path]
        dist.broadcast_object_list(remote_path_list, src=0)
        remote_path = remote_path_list[0]

    def _get_tmp_dir(self):
        return remote_path

    fork_context = multiprocessing.get_context('fork')
    with patch('composer.utils.object_store.utils.S3ObjectStore', DummyObjectStore):
        with patch('tests.utils.test_remote_uploader.DummyObjectStore.get_tmp_dir', _get_tmp_dir):
            with patch('composer.utils.remote_uploader.multiprocessing.get_context', lambda _: fork_context):
                symlink_file_name = 'latest.symlink'
                engine = None
                if async_upload:
                    logger = Logger(dummy_state)
                    engine = Engine(state=dummy_state, logger=logger)
                path = 'save_path'
                upload_file(
                    source_path=local_file_full_name,
                    dest_dir='S3://bucket_name/' + path,
                    symlink_granularity='file',
                    async_upload=async_upload,
                    state=dummy_state,
                )
                if async_upload:
                    assert engine is not None
                    engine.run_event(Event.FIT_END)
                if world_size > 1:
                    dist.barrier()

                remote_file_name = os.path.join(remote_path.name, path, file_name)
                assert os.path.isfile(remote_file_name)
                with open(remote_file_name, 'r') as f:
                    assert f.read() == str(rank)
                remote_symlink_file_name = os.path.join(remote_path.name, path, symlink_file_name)
                if rank == 0:
                    assert os.path.isfile(remote_symlink_file_name)
                    with open(remote_symlink_file_name, 'r') as f:
                        assert f.read() == path
                if world_size > 1:
                    dist.barrier()
