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

@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('use_cpu', [False, True])
@pytest.mark.parametrize('sharded_model', [False, True])
def test_save_full_state_dict_to_disk(world_size: int, tmp_path: str, use_cpu: bool, sharded_model: bool):
    if sharded_model and use_cpu:
        pytest.skip("sharded_model is only for GPU")
    
    tmp_path = dist.all_gather_object(tmp_path)[0]

    destination_file_path = os.path.join(tmp_path, 'test.pt')
    use_fsdp = sharded_model
    model, _ = _init_model(use_fsdp=use_fsdp, device='cpu' if use_cpu else 'cuda')
    state_dict = get_model_state_dict(model, sharded_state_dict=False)
    path_saved = save_state_dict_to_disk(state_dict, destination_file_path=destination_file_path)
    time.sleep(1)
    if dist.get_global_rank() == 0:
        assert path_saved == destination_file_path
        assert os.path.exists(destination_file_path), f'{destination_file_path} does not exist'
        loaded_state_dict = torch.load(path_saved, map_location='cpu' if use_cpu else 'cuda')
        deep_compare(state_dict, loaded_state_dict)
    else:
        assert path_saved is None


def test_save_sharded_state_dict_to_disk():
    pass

def test_save_hybrid_sharded_state_dict_to_disk():
    pass




