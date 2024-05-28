# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import pytest
import torch
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from composer.checkpoint import get_metadata_state_dict, get_model_state_dict
from composer.utils import dist
from tests.common.compare import deep_compare
from tests.common.markers import world_size
from tests.common.models import EvenSimplerMLP, SimpleComposerMLP, configure_tiny_gpt2_hf_model


@pytest.mark.gpu
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_unsharded_model(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=8, device='cuda')
    model_state_dict = get_model_state_dict(model, sharded_state_dict=False, include_keys=None, ignore_keys=None)
    for name, param in model.named_parameters():
        print(name)
        assert name in model_state_dict
        assert torch.equal(model_state_dict[name], param)


@pytest.mark.gpu
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_include(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=8, device='cuda')
    model_state_dict = get_model_state_dict(model, sharded_state_dict=False, include_keys=['module.0.weight'])
    assert set(model_state_dict.keys()) == {'module.0.weight'}

    model_state_dict = get_model_state_dict(model, sharded_state_dict=False, include_keys='module.2*')
    assert set(model_state_dict.keys()) == {'module.2.weight'}


@pytest.mark.gpu
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_ignore(use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=8, device='cuda')

    model_state_dict = get_model_state_dict(model, sharded_state_dict=False, ignore_keys='module.2.weight')
    assert set(model_state_dict.keys()) == {'module.0.weight'}

    model_state_dict = get_model_state_dict(model, sharded_state_dict=False, ignore_keys=['module.2*'])
    assert set(model_state_dict.keys()) == {'module.0.weight'}


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_full_for_sharded_model(world_size, tensor_type, use_composer_model: bool):
    if tensor_type == 'dtensor' and version.parse(torch.__version__) < version.parse('2.2.0'):
        pytest.skip('DTensor is only supported in PyTorch >= 2.2.0')
    if use_composer_model:
        model = SimpleComposerMLP(num_features=16, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=16, device='cuda')

    # Torch flattens model params in place after wrapped with FSDP, so we need to cache unflattened params now
    # before fsdp wrapping in order to keep pre-sharding shapes.
    pre_shard_state_dict = get_model_state_dict(
        model,
        sharded_state_dict=False,
    )
    fsdp_kwargs: Dict[str, Any] = dict(
        use_orig_params=True,
        sync_module_states=True,  # To enable easy comparison between rank 0 unsharded model and full state dict
    )

    if tensor_type == 'dtensor':
        from torch.distributed.device_mesh import init_device_mesh
        device_mesh = init_device_mesh('cuda', (2,))
        fsdp_kwargs['device_mesh'] = device_mesh

    sharded_model = FSDP(
        model,
        **fsdp_kwargs,
    )

    post_shard_full_state_dict = get_model_state_dict(sharded_model, sharded_state_dict=False)

    if dist.get_global_rank() == 0:
        deep_compare(pre_shard_state_dict, post_shard_full_state_dict)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_sharded(world_size, tensor_type, use_composer_model: bool):
    if tensor_type == 'dtensor' and version.parse(torch.__version__) < version.parse('2.2.0'):
        pytest.skip('DTensor is only supported in PyTorch >= 2.2.0')

    if use_composer_model:
        model = SimpleComposerMLP(num_features=16, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=16, device='cuda')

    # Torch flattens model params in place after wrapped with FSDP, so we need to cache unflattened params now
    # before fsdp wrapping in order to keep pre-sharding shapes.
    pre_shard_full_state_dict = get_model_state_dict(
        model,
        sharded_state_dict=False,
    )

    fsdp_kwargs: Dict[str, Any] = dict(
        use_orig_params=True,
        sync_module_states=True,  # To enable easy comparison between rank 0 unsharded model and full state dict
    )

    if tensor_type == 'dtensor':
        from torch.distributed.device_mesh import init_device_mesh
        device_mesh = init_device_mesh('cuda', (2,))
        fsdp_kwargs.update(device_mesh=device_mesh)

    sharded_model = FSDP(
        model,
        **fsdp_kwargs,
    )

    post_shard_sharded_sd = get_model_state_dict(sharded_model, sharded_state_dict=True)

    # In order to test if the sharded state dict is correct we go through this process:
    # 1. Transform the each rank's state dict's values by extracting the the local tensor from the ShardedTensor object
    # 2. Gather each rank's state dicts
    # 3. Make a "reconstructed" full state dict by, for each key, concatenating all the tensor shards into one big tensor
    # 4. Compare this "reconstructed" full state dict to the original model's state dict to ensure they are the same.
    local_tensor_sd = {
        n: (p.local_tensor() if tensor_type == 'sharded_tensor' else p.to_local())
        for n, p in post_shard_sharded_sd.items()
    }
    all_local_tensor_sd = dist.all_gather_object(local_tensor_sd)
    post_shard_reconstructed_full_sd = {
        n: torch.cat(
            [sd[n].cuda() for sd in all_local_tensor_sd],
            dim=0,  # dim=0 because fsdp shards each tensor on the 0th dimension
        ) for n in pre_shard_full_state_dict.keys()
    }
    if dist.get_global_rank() == 0:
        deep_compare(pre_shard_full_state_dict, post_shard_reconstructed_full_sd)


@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize(
    'precision',
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_precision_sharded_model(
    world_size,
    tensor_type,
    precision: str,
    use_composer_model: bool,
):
    if tensor_type == 'dtensor' and version.parse(torch.__version__) < version.parse('2.2.0'):
        pytest.skip('DTensor is only supported in PyTorch >= 2.2.0')
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=8, device='cuda')

    fsdp_kwargs: Dict[str, Any] = dict(
        use_orig_params=True,
        sync_module_states=True,  # To enable easy comparison between rank 0 unsharded model and full state dict
    )

    if tensor_type == 'dtensor':
        from torch.distributed.device_mesh import init_device_mesh
        device_mesh = init_device_mesh('cuda', (2,))
        fsdp_kwargs.update(device_mesh=device_mesh)

    sharded_model = FSDP(
        model,
        **fsdp_kwargs,
    )

    model_state_dict = get_model_state_dict(
        sharded_model,
        precision=precision,
        sharded_state_dict=True,
        include_keys=None,
        ignore_keys=None,
    )
    for sharded_tens in model_state_dict.values():
        local_tensor = sharded_tens.local_tensor() if tensor_type == 'sharded_tensor' else sharded_tens.to_local()
        assert local_tensor.dtype == precision


@pytest.mark.gpu
@pytest.mark.parametrize(
    'precision',
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_model_state_dict_precision_unsharded_model(precision: str, use_composer_model: bool):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=8, device='cuda')
    else:
        model = EvenSimplerMLP(num_features=8, device='cuda')
    model_state_dict = get_model_state_dict(
        model,
        precision=precision,
        sharded_state_dict=False,
        include_keys=None,
        ignore_keys=None,
    )
    for tens in model_state_dict.values():
        assert tens.dtype == precision


@pytest.mark.gpu
@world_size(1, 2)
def test_get_metadata_empty_call(world_size):
    metadata_sd = get_metadata_state_dict()
    for key in [
        'composer_version',
        'composer_commit_hash',
        'torch_version',
        'python_version',
        'num_nodes',
        'num_gpus_per_node',
        'num_gpus',
        'gpu_model',
        'cpu_model',
        'cpu_core_count',
    ]:
        assert key in metadata_sd

    assert metadata_sd['num_nodes'] == 1
    assert metadata_sd['num_gpus'] == world_size


@pytest.mark.gpu
@pytest.mark.parametrize('model_type', ['composer', 'hf', 'nn.module'])
def test_get_metadata_unsharded_model(model_type: str):
    if model_type == 'composer':
        model = SimpleComposerMLP(num_features=8, device='cuda')
        expected_model_name = 'SimpleComposerMLP'
    elif model_type == 'nn.module':
        model = EvenSimplerMLP(num_features=8, device='cuda')
        expected_model_name = 'EvenSimplerMLP'
    else:
        model = configure_tiny_gpt2_hf_model()
        expected_model_name = 'GPT2LMHeadModel'

    generate_parameter_info = False if model_type == 'hf' else True
    metadata_sd = get_metadata_state_dict(model, generate_parameter_info=generate_parameter_info)
    assert metadata_sd['model_name'] == expected_model_name
    if model_type == 'hf':
        assert 'huggingface' in metadata_sd
        assert 'model' in metadata_sd['huggingface']
        assert 'tokenizer' in metadata_sd['huggingface']
        assert 'model_name' in metadata_sd
    else:
        assert 'parameter_info' in metadata_sd
        assert metadata_sd['parameter_info']['module.0.weight'] == {'shape': (8, 8), 'requires_grad': True}
        assert metadata_sd['parameter_info']['module.2.weight'] == {'shape': (8, 8), 'requires_grad': True}
