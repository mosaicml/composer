from composer.checkpoint.save import save_state_dict_to_disk
from composer.checkpoint.state_dict import get_model_state_dict
from tests.common.markers import world_size
from tests.common.compare import deep_compare
from tests.checkpoint import _init_model
import os
from composer.utils import dist
import torch
import pytest
import time
from composer.utils.checkpoint import _TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME
from copy import deepcopy
import torch.distributed.checkpoint as DCP
from pathlib import Path
import uuid

@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('sharded_model', [False, True])
def test_save_full_state_dict_to_disk(world_size: int, tmp_path: str, sharded_model: bool):
        

    destination_file_path = os.path.join(tmp_path, 'test.pt')
    use_fsdp = sharded_model
    model, _ = _init_model(use_fsdp=use_fsdp, device='cuda',
                           sync_module_states=True)

    state_dict = get_model_state_dict(model, sharded_state_dict=False)
    path_saved = save_state_dict_to_disk(state_dict, destination_file_path=destination_file_path)
    time.sleep(1)
    if dist.get_global_rank() == 0:
        assert path_saved == destination_file_path
        assert os.path.exists(destination_file_path), f'{destination_file_path} does not exist'
        loaded_state_dict = torch.load(path_saved, map_location='cuda')
        deep_compare(state_dict, loaded_state_dict)
    else:
        assert path_saved is None

@world_size(2)
@pytest.mark.filterwarnings('ignore:The passed') # Torch issues a warning for wrapping a CPU model in FSDP
@pytest.mark.parametrize('sharded_model', [False, True])
def test_save_full_state_dict_to_disk_cpu(world_size: int, tmp_path: str, sharded_model: bool):
    #tmp_path = dist.all_gather_object(tmp_path)[0]

    destination_file_path = os.path.join(tmp_path, 'test.pt')
    use_fsdp = sharded_model
    model, _ = _init_model(use_fsdp=use_fsdp, device='cpu', sync_module_states=False, cpu_offload=True)
    state_dict = get_model_state_dict(model, sharded_state_dict=False)
    path_saved = save_state_dict_to_disk(state_dict, destination_file_path=destination_file_path)
    time.sleep(1)
    if dist.get_global_rank() == 0:
        assert path_saved == destination_file_path
        assert os.path.exists(destination_file_path), f'{destination_file_path} does not exist'
        loaded_state_dict = torch.load(path_saved, map_location='cpu')
        deep_compare(state_dict, loaded_state_dict)
    else:
        assert path_saved is None

@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
def test_save_sharded_state_dict_to_disk(world_size: int, tmp_path: str, tensor_type: str):
    destination_file_path = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    # Sync the path across all ranks
    destination_file_path = dist.all_gather_object(destination_file_path)[0]
    model, _ = _init_model(use_fsdp=True, device='cuda', tensor_type=tensor_type)

    state_dict = get_model_state_dict(model, sharded_state_dict=True)
    loaded_in_state_dict = deepcopy(state_dict)
    path_saved = save_state_dict_to_disk(state_dict, 
                                         destination_file_path=destination_file_path,
                                         overwrite=True)
    assert path_saved == f"{destination_file_path}/{_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME}"
    
    load_path = str(Path(path_saved).parent)
    DCP.load(state_dict=loaded_in_state_dict,
             checkpoint_id=load_path)
    deep_compare(state_dict, loaded_in_state_dict)

@pytest.mark.filterwarnings('ignore:The passed') # Torch issues a warning for wrapping a CPU model in FSDP
@world_size(2)
def test_save_sharded_state_dict_to_disk_cpu(world_size: int, tmp_path: str):
    destination_file_path = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_file_path = dist.all_gather_object(destination_file_path)[0]
    model, _ = _init_model(use_fsdp=True, device='cpu', sync_module_states=False, cpu_offload=True)
    state_dict = get_model_state_dict(model, sharded_state_dict=True)
    loaded_in_state_dict = deepcopy(state_dict)
    path_saved = save_state_dict_to_disk(state_dict, destination_file_path=destination_file_path,
                                         overwrite=True)
    assert path_saved == f"{destination_file_path}/{_TORCH_DISTRIBUTED_CHECKPOINTS_FILENAME}"

    load_path = str(Path(path_saved).parent)
    DCP.load(state_dict=loaded_in_state_dict,
             checkpoint_id=load_path)
    deep_compare(state_dict, loaded_in_state_dict)
