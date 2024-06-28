# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import tempfile
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed.checkpoint as DCP

from composer.utils import dist
from tests.checkpoint.helpers import init_model
from tests.common.markers import world_size

class TestCheckpointDownload:
    def setup(self):
        self.remote_dir = tempfile.TemporaryDirectory()
        self.fsdp_model, _ = init_model(
            use_fsdp=True,
        )

    def save_monolithic_checkpoint(self):
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT):
            state = fsdp.state_dict()
        if dist.get_global_ran() != 0:
            return
        
        writer = DCP.FileSystemWriter(path=self.remote_dir)
        if version.parse(torch.__version__) < version.parse('2.2.0'):
            DCP.save_state_dict(state_dict=state, storage_writer=writer)
        else:
            DCP.save(state_dict=state, storage_writer=writer)

    @world_size(1, 2)
    @pytest.mark.gpu
    def test_download_monolithid_checkpoint(self):
        self.setup()
        self.save_monolithic_checkpoint()

    def test_download_sharded_checkpoint(self):
        pass
