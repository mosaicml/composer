# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as DCP
from packaging import version

from composer.checkpoint.save import (
    save_checkpoint_to_disk,
    save_composer_metadata_to_disk,
    save_model_to_disk,
    save_optim_to_disk,
    save_state_dict_to_disk,
)
from composer.checkpoint.state_dict import get_model_state_dict, get_optim_state_dict
from composer.core import Timestamp
from composer.utils import dist
from composer.utils.checkpoint import (
    _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME,
    _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME,
)
from tests.checkpoint.helpers import init_model, init_model_and_optimizer, init_state
from tests.common.compare import deep_compare
from tests.common.markers import world_size


@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_model,sharded_checkpoint',
    [
        pytest.param(1, False, False, marks=pytest.mark.world_size(1)),
        pytest.param(2, True, True, marks=pytest.mark.world_size(2)),
        pytest.param(2, True, False, marks=pytest.mark.world_size(2)),
    ],
)
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_save_checkpoint_to_disk(world_size: int, tmp_path: str, sharded_model: bool, sharded_checkpoint: bool):
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]
    save_options = {
        'save_model': True,
        'save_optimizer': True,
        'save_resumption_state': True,
        'sharded_checkpoint': sharded_checkpoint,
        'dir_prefix': 'ep{epoch}-ba{batch}',
    }
    state = init_state(use_fsdp=sharded_model, device='cuda', take_step=True)
    state.run_name = 'foo'
    state.timestamp = Timestamp()
    expected_destination_dir = os.path.join(destination_dir, 'ep0-ba0')
    save_checkpoint_to_disk(destination_dir=destination_dir, state=state, options=save_options)
    expected_model_dir = os.path.join(expected_destination_dir, 'model')
    expected_optim_dir = os.path.join(expected_destination_dir, 'optim')
    expected_metadata_filepath = os.path.join(expected_destination_dir, 'composer_metadata.json')
    expected_resumption_filepath = os.path.join(expected_destination_dir, 'resumption.pkl')
    if sharded_checkpoint:
        checkpoint_filenames = dist.all_gather_object(_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME)
        for checkpoint_filename in checkpoint_filenames:
            assert os.path.exists(os.path.join(expected_model_dir, checkpoint_filename))
            assert os.path.exists(os.path.join(expected_optim_dir, checkpoint_filename))
        assert os.path.exists(os.path.join(expected_model_dir, _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME))
        assert os.path.exists(os.path.join(expected_optim_dir, _TORCH_DISTRIBUTED_CHECKPOINTS_METADATA_FILENAME))
    else:
        assert os.path.exists(os.path.join(expected_model_dir, 'model.pt'))
        assert os.path.exists(os.path.join(expected_optim_dir, 'optim.pt'))

    import time

    # Need to wait for the file to be written to avoid flaky test.
    time.sleep(0.2)
    assert os.path.exists(expected_metadata_filepath)
    assert os.path.exists(expected_resumption_filepath)


def test_save_composer_metadata_to_disk(tmp_path: str):
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]
    destination_dir = os.path.join(destination_dir, 'composer_metadata.json')
    save_composer_metadata_to_disk(destination_dir)
    expected_file_path = destination_dir
    assert os.path.exists(expected_file_path)
    json.load(open(expected_file_path, 'r'))


@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_optimizer,sharded_checkpoint',
    [
        pytest.param(1, False, False, marks=pytest.mark.world_size(1)),
        pytest.param(2, True, True, marks=pytest.mark.world_size(2)),
        pytest.param(2, True, False, marks=pytest.mark.world_size(2)),
    ],
)
def test_save_optim_to_disk(world_size: int, tmp_path: str, sharded_optimizer: bool, sharded_checkpoint: bool):
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    # Sync the path across all ranks
    destination_dir = dist.all_gather_object(destination_dir)[0]
    destination_dir = os.path.join(destination_dir, 'optim.pt') if not sharded_checkpoint else destination_dir
    use_fsdp = sharded_optimizer
    model, optim = init_model_and_optimizer(use_fsdp=use_fsdp, device='cuda')
    optim_state_dict = get_optim_state_dict(model, optimizer=optim, sharded_state_dict=sharded_checkpoint)
    optim_state_dict_saved = deepcopy(optim_state_dict)
    save_optim_to_disk(model, optim, destination_dir=destination_dir, sharded_checkpoint=sharded_checkpoint)

    # Load new optim from disk
    model, optim = init_model_and_optimizer(use_fsdp=use_fsdp, device='cuda')
    cur_state_dict = get_optim_state_dict(model, optimizer=optim, sharded_state_dict=sharded_checkpoint)

    if sharded_checkpoint:
        expected_file_path = os.path.join(destination_dir)
        if version.parse(torch.__version__) < version.parse('2.2.0'):
            DCP.load_state_dict(state_dict=cur_state_dict, storage_reader=DCP.FileSystemReader(expected_file_path))
        else:
            DCP.load(state_dict=cur_state_dict, storage_reader=DCP.FileSystemReader(expected_file_path))
    else:
        if dist.get_global_rank() == 0:
            expected_file_path = destination_dir
            cur_state_dict = torch.load(expected_file_path, map_location='cuda')

    deep_compare(optim_state_dict_saved, cur_state_dict)


@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_model,sharded_checkpoint',
    [
        pytest.param(1, False, False, marks=pytest.mark.world_size(1)),
        pytest.param(2, True, True, marks=pytest.mark.world_size(2)),
        pytest.param(2, True, False, marks=pytest.mark.world_size(2)),
    ],
)
def test_save_model_to_disk(world_size: int, tmp_path: str, sharded_model: bool, sharded_checkpoint: bool):
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    # Sync the path across all ranks
    destination_dir = dist.all_gather_object(destination_dir)[0]
    use_fsdp = sharded_model
    model, _ = init_model(use_fsdp=use_fsdp, device='cuda', sync_module_states=True)
    state_dict = get_model_state_dict(model, sharded_state_dict=sharded_checkpoint)
    state_dict_saved = deepcopy(state_dict)
    destination_dir = os.path.join(destination_dir, 'model.pt') if not sharded_checkpoint else destination_dir
    save_model_to_disk(model, destination_dir=destination_dir, sharded_checkpoint=sharded_checkpoint)

    # Load new model from disk
    new_model, _ = init_model(use_fsdp=use_fsdp, device='cuda', sync_module_states=True)
    cur_state_dict = get_model_state_dict(new_model, sharded_state_dict=sharded_checkpoint)

    if sharded_checkpoint:
        expected_file_path = destination_dir
        if version.parse(torch.__version__) < version.parse('2.2.0'):
            DCP.load_state_dict(state_dict=cur_state_dict, storage_reader=DCP.FileSystemReader(expected_file_path))
        else:
            DCP.load(state_dict=cur_state_dict, storage_reader=DCP.FileSystemReader(expected_file_path))
    else:
        if dist.get_global_rank() == 0:
            expected_file_path = destination_dir
            cur_state_dict = torch.load(expected_file_path, map_location='cuda')

    deep_compare(state_dict_saved, cur_state_dict)


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
