# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from composer.checkpoint import download_monolithic_checkpoint
from composer.utils import dist
from tests.checkpoint.helpers import init_model
from tests.common.markers import world_size
from tests.utils.test_remote_uploader import DummyObjectStore


@world_size(1, 2)
@pytest.mark.gpu
@pytest.mark.parametrize('rank_zero_only', [True, False])
def test_download_monolithic_checkpoint(world_size: int, rank_zero_only: bool):
    # Write a checkpoint
    tmp_dir = tempfile.TemporaryDirectory()
    use_fsdp = False
    if world_size > 1:
        use_fsdp = True
    fsdp_model, _ = init_model(use_fsdp=use_fsdp)

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    state = get_model_state_dict(fsdp_model, options=StateDictOptions(full_state_dict=True))

    checkpoint_filename = 'state_dict'
    save_filename = os.path.join(tmp_dir.name, checkpoint_filename)
    if dist.get_global_rank() == 0:
        torch.save(state, save_filename)

    class DummyS3ObjectStore(DummyObjectStore):

        def get_tmp_dir(self):
            return tmp_dir

    # Download a monolithic checkpoint
    local_file_name = 'state_dict.download'
    with patch('composer.utils.file_helpers.S3ObjectStore', DummyS3ObjectStore):
        ret = download_monolithic_checkpoint(
            source_uri=f's3://bucket_name/{checkpoint_filename}',
            destination_path=local_file_name,
            global_rank_zero_only=rank_zero_only,
        )
    dist.barrier()

    if rank_zero_only and dist.get_global_rank() != 0:
        assert ret == None
    if dist.get_global_rank() == 0:
        assert ret == local_file_name
        assert os.path.isfile(local_file_name) == True
