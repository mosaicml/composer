# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from composer.checkpoint.state_dict import get_model_state_dict
from tests.common.models import EvenSimplerMLP, SimpleComposerMLP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tests.common.markers import world_size
from composer.utils import dist
from tests.common.compare import deep_compare


@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_unsharded_model(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cpu')
    else:
        model = EvenSimplerMLP(num_features=8, device='cpu')
    model_state_dict = get_model_state_dict(model, sharded=False, include_keys=None, ignore_keys=None)
    for name, param in model.named_parameters():
        print(name)
        assert name in model_state_dict
        assert torch.equal(model_state_dict[name], param)


@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_include(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cpu')
    else:
        model = EvenSimplerMLP(num_features=8, device='cpu')
    model_state_dict = get_model_state_dict(model, sharded=False, include_keys=['module.0.weight'])
    assert set(model_state_dict.keys()) == {'module.0.weight'}

    model_state_dict = get_model_state_dict(model, sharded=False, include_keys='module.2*')
    assert set(model_state_dict.keys()) == {'module.2.weight'}


@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_ignore(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cpu')
    else:
        model = EvenSimplerMLP(num_features=8, device='cpu')

    model_state_dict = get_model_state_dict(model, sharded=False, ignore_keys='module.2.weight')
    assert set(model_state_dict.keys()) == {'module.0.weight'}

    model_state_dict = get_model_state_dict(model, sharded=False, ignore_keys=['module.2*'])
    assert set(model_state_dict.keys()) == {'module.0.weight'}


#TODO add tests for sharded and for precision
@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_full_for_sharded_model(world_size, use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=16, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=16, device='cuda')

    # Torch flattens model params in place after wrapped with FSDP, so we need to cache unflattened params now
    # before fsdp wrapping in order to keep pre-sharding shapes.
    pre_shard_state_dict = get_model_state_dict(model, 
                                                sharded=False,
                                                cpu_offload=True, # Set this to True, so that both state dicts will be on cpu
                                                )

    sharded_model = FSDP(model,
                         use_orig_params=True, 
                         sync_module_states=True, # We set this to enable easy comparison between rank 0 unsharded model and full state dict
                         )

    post_shard_full_state_dict = get_model_state_dict(sharded_model, sharded=False)

    if dist.get_global_rank() == 0:
        deep_compare(pre_shard_state_dict, post_shard_full_state_dict) 


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_sharded(world_size, use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=16, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=16, device='cuda')
    
    # Torch flattens model params in place after wrapped with FSDP, so we need to cache unflattened params now
    # before fsdp wrapping in order to keep pre-sharding shapes.
    pre_shard_full_state_dict = get_model_state_dict(model, 
                                            sharded=False,
                                            cpu_offload=True, # Set this to True, so that both state dicts will be on cpu
                                            )
    sharded_model = FSDP(model,
                        use_orig_params=True,
                        sync_module_states=True
                        )
    
    post_shard_sharded_sd = get_model_state_dict(sharded_model, sharded=True)

    # In order to test if the sharded state dict is correct we go through this process:
    # 1. Transform the each rank's state dict's values by extracting the the local tensor from the ShardedTensor object
    # 2. Gather each rank's state dicts
    # 3. Make a "reconstructed" full state dict by, for each key, concatenating all the tensor shards into one big tensor
    # 4. Compare this "reconstructed" full state dict to the original model's state dict to ensure they are the same.
    local_tensor_sd = {n: p.local_tensor() for n,p in post_shard_sharded_sd.items()}
    all_local_tensor_sd = dist.all_gather_object(local_tensor_sd)
    post_shard_reconstructed_full_sd = {n: torch.cat([sd[n] for sd in all_local_tensor_sd], 
                                                     dim=0, # dim=0 because fsdp shards each tensor on the 0th dimension
                                                     ) 
                                                     for n in pre_shard_full_state_dict.keys()}
    if dist.get_global_rank() == 0:
        deep_compare(pre_shard_full_state_dict, post_shard_reconstructed_full_sd)
    
    