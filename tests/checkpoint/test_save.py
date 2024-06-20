# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import time
import uuid
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as DCP
from packaging import version

from composer.checkpoint.save import save_state_dict_to_disk
from composer.checkpoint.state_dict import get_model_state_dict
from composer.utils import dist
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
from tests.checkpoint.helpers import init_model
from tests.common.compare import deep_compare
from tests.common.markers import world_size


@world_size(1, 2)
@pytest.mark.gpu
@pytest.mark.parametrize('sharded_model', [False, True])
def test_save_full_state_dict_to_disk(world_size: int, tmp_path: str, sharded_model: bool):
    if world_size == 1 and sharded_model:
        pytest.skip("Can't have a sharded model for world_size = 1")
    destination_file_path = os.path.join(tmp_path, 'test.pt')
    use_fsdp = sharded_model
    model, _ = init_model(use_fsdp=use_fsdp, device='cuda', sync_module_states=True)

    state_dict = get_model_state_dict(model, sharded_state_dict=False)
    path_saved = save_state_dict_to_disk(state_dict, destination_file_path=destination_file_path)
    time.sleep(1)
    if dist.get_global_rank() == 0:
        assert path_saved is not None
        assert path_saved == destination_file_path
        assert os.path.exists(destination_file_path), f'{destination_file_path} does not exist'
        loaded_state_dict = torch.load(path_saved, map_location='cuda')
        deep_compare(state_dict, loaded_state_dict)
    else:
        assert path_saved is None


@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize(
    'tensor_type',
    [
        'sharded_tensor',
        pytest.param(
            'dtensor',
            marks=pytest.mark.skipif(
                version.parse(torch.__version__) < version.parse('2.2.0'),
                reason='Requires torch>=2.2.0 for dtensor',
            ),
        ),
    ],
)
def test_save_sharded_state_dict_to_disk(world_size: int, tmp_path: str, tensor_type: str):

    destination_file_path = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    # Sync the path across all ranks
    destination_file_path = dist.all_gather_object(destination_file_path)[0]
    model, _ = init_model(use_fsdp=True, device='cuda', tensor_type=tensor_type)

    state_dict = get_model_state_dict(model, sharded_state_dict=True)
    loaded_in_state_dict = deepcopy(state_dict)
    path_saved = save_state_dict_to_disk(state_dict, destination_file_path=destination_file_path, overwrite=True)
    assert path_saved == f'{destination_file_path}/{_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME}'
    assert path_saved is not None
    load_path = str(Path(path_saved).parent)
    if version.parse(torch.__version__) < version.parse('2.2.0'):
        DCP.load_state_dict(state_dict=loaded_in_state_dict, storage_reader=DCP.FileSystemReader(load_path))
    else:
        DCP.load(state_dict=loaded_in_state_dict, storage_reader=DCP.FileSystemReader(load_path))
    deep_compare(state_dict, loaded_in_state_dict)
