# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import pytest
import torch
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import adam

from composer.checkpoint import get_metadata_state_dict, get_model_state_dict, get_optim_state_dict
from composer.devices import DeviceGPU
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
    fsdp_kwargs: dict[str, Any] = dict(
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

    fsdp_kwargs: dict[str, Any] = dict(
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

    fsdp_kwargs: dict[str, Any] = dict(
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


def _init_model_and_optimizer(
    use_composer_model: bool,
    num_classes=3,
    batch_size=5,
    num_features=8,
    take_step=True,
    use_fsdp=False,
    tensor_type='sharded_tensor',
):
    model, loss_fn = _init_model(
        use_composer_model,
        num_classes=num_classes,
        batch_size=batch_size,
        num_features=num_features,
        use_fsdp=use_fsdp,
        tensor_type=tensor_type,
    )

    optimizer = _init_optimizer(
        model,
        loss_fn,
        use_composer_model=use_composer_model,
        num_classes=num_classes,
        batch_size=batch_size,
        num_features=num_features,
        take_step=take_step,
    )

    return model, optimizer


def _init_model(
    use_composer_model: bool = False,
    num_classes=3,
    batch_size=5,
    num_features=8,
    use_fsdp=False,
    tensor_type='sharded_tensor',
):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=num_features, num_classes=num_classes, device='cuda')
        loss_fn = model._loss_fn
    else:
        model = EvenSimplerMLP(num_features=num_features, num_out_features=num_classes, device='cuda')
        loss_fn = torch.nn.CrossEntropyLoss()

    if use_fsdp:
        fsdp_kwargs: Dict[str, Any] = dict(
            use_orig_params=True,
            sync_module_states=True,  # To enable easy comparison between rank 0 unsharded model and full state dict
        )

        if tensor_type == 'dtensor':
            from torch.distributed.device_mesh import init_device_mesh
            device_mesh = init_device_mesh('cuda', (2,))
            fsdp_kwargs['device_mesh'] = device_mesh

        model = FSDP(
            model,
            **fsdp_kwargs,
        )

    return model, loss_fn


def _init_optimizer(
    model,
    loss_fn,
    use_composer_model: bool = False,
    num_classes=3,
    batch_size=5,
    num_features=8,
    take_step=True,
):
    inputs = torch.randn(batch_size, num_features, device='cuda')
    targets = torch.randint(low=0, high=num_classes, size=(batch_size,), device='cuda', dtype=torch.long)
    batch = (inputs, targets) if use_composer_model else inputs
    optimizer = adam.Adam(model.parameters())
    outputs = model(batch)
    loss = loss_fn(outputs, targets)
    loss.backward()
    if take_step:
        optimizer.step()
    return optimizer


@pytest.mark.gpu
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_optim_state_dict_unsharded_model(use_composer_model: bool):
    model, optimizer = _init_model_and_optimizer(use_composer_model=use_composer_model, take_step=True)
    optim_state_dict = get_optim_state_dict(model, optimizer)

    # Dict mapping parameter index to optimizer state for that parameter.
    osd_state = optim_state_dict['state']
    # Dict mapping parameter itself to optimizer state for that parameter.
    optim_state = optimizer.state

    # Make sure optimizer state is the same between the state dict and the optimizer object.
    for osd_param_state, opt_param_state in zip(osd_state.values(), optim_state.values()):
        deep_compare(osd_param_state, opt_param_state)

    # Make sure the optimizer state in the state dict is the same shape as the parameter it corresponds to.
    # Because model is unsharded the optimizer state should have keys corresponding to the index of the model's parameters.
    # e.g. if the model has 3 parameters, the optimizer state dict keys would be (0,1,2).
    params = list(model.parameters())
    param_dict = dict(list(model.named_parameters()))
    for param_key, param_state in osd_state.items():
        if isinstance(param_key, str):
            param = param_dict[param_key]
        else:
            param = params[param_key]
        assert param.shape == param_state['exp_avg'].shape
        assert param.shape == param_state['exp_avg_sq'].shape

    # Make sure param groups between the state dict and the optimizer object are the same.
    for osd_group, opt_group in zip(optim_state_dict['param_groups'], optimizer.param_groups):
        # Only params should differ between the two.
        # * in the optimizer state dict params will be indices into the model's parameters list.
        # * in the optimizer object params will be the actual parameter tensors.
        deep_compare(osd_group, opt_group, ignore_keys=['params'])


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
def test_get_optim_state_dict_precision_unsharded_model(precision: str, use_composer_model: bool):
    model, optimizer = _init_model_and_optimizer(use_composer_model=use_composer_model, take_step=True)
    optim_state_dict = get_optim_state_dict(model, optimizer, precision=precision)
    for param_state in optim_state_dict['state'].values():
        assert param_state['exp_avg'].dtype == precision
        assert param_state['exp_avg_sq'].dtype == precision


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_optim_dict_full_for_sharded_model(world_size, tensor_type, use_composer_model: bool):
    if tensor_type == 'dtensor' and version.parse(torch.__version__) < version.parse('2.2.0'):
        pytest.skip('DTensor is only supported in PyTorch >= 2.2.0')

    model, optimizer = _init_model_and_optimizer(
        use_composer_model=use_composer_model,
        take_step=True,
        use_fsdp=True,
        tensor_type=tensor_type,
    )
    optim_state_dict = get_optim_state_dict(model, optimizer, sharded_state_dict=False)

    with FSDP.summon_full_params(model):
        # Make sure the optimizer state in the state dict is the same shape as the parameter it corresponds to.
        fqn_to_shape_map = {fqn: param.shape for fqn, param in model.named_parameters()}
        if dist.get_global_rank() == 0:
            # Because model is sharded, the state dict should have the same keys as the model's parameters.
            for fqn, param_state in optim_state_dict['state'].items():
                model_param_shape = fqn_to_shape_map[fqn]
                assert model_param_shape == param_state['exp_avg'].shape
                assert model_param_shape == param_state['exp_avg_sq'].shape


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_optim_dict_sharded_for_sharded_model(world_size, tensor_type, use_composer_model: bool):
    if tensor_type == 'dtensor' and version.parse(torch.__version__) < version.parse('2.2.0'):
        pytest.skip('DTensor is only supported in PyTorch >= 2.2.0')

    model, optimizer = _init_model_and_optimizer(
        use_composer_model=use_composer_model,
        take_step=True,
        use_fsdp=True,
        tensor_type=tensor_type,
    )
    model_state_dict = get_model_state_dict(model, sharded_state_dict=True)
    optim_state_dict = get_optim_state_dict(model, optimizer, sharded_state_dict=True)

    # Check to make sure on every rank optimizer state name and shape matches model's
    fqn_to_shape_map = {fqn: param.shape for fqn, param in model_state_dict.items()}
    for fqn, param_state in optim_state_dict['state'].items():
        model_param_shape = fqn_to_shape_map[fqn]
        assert model_param_shape == param_state['exp_avg'].shape
        assert model_param_shape == param_state['exp_avg_sq'].shape


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

    metadata_sd = get_metadata_state_dict(model)
    assert metadata_sd['model_name'] == expected_model_name
    if model_type == 'hf':
        assert 'huggingface' in metadata_sd
        assert 'model' in metadata_sd['huggingface']
        assert 'tokenizer' in metadata_sd['huggingface']
        assert 'model_name' in metadata_sd


@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('tensor_type', ['sharded_tensor', 'dtensor'])
@pytest.mark.parametrize('model_type', ['composer', 'hf', 'nn.module'])
def test_get_metadata_sharded_model(model_type: str, tensor_type: str, world_size: int):
    if tensor_type == 'dtensor' and version.parse(torch.__version__) < version.parse('2.2.0'):
        pytest.skip('DTensor is only supported in PyTorch >= 2.2.0')
    if model_type == 'composer':
        model = SimpleComposerMLP(num_features=8, device='cuda')
        expected_model_name = 'SimpleComposerMLP'
    elif model_type == 'nn.module':
        model = EvenSimplerMLP(num_features=8, device='cuda')
        expected_model_name = 'EvenSimplerMLP'
    else:
        model = configure_tiny_gpt2_hf_model().cuda()
        expected_model_name = 'GPT2LMHeadModel'

    fsdp_kwargs: dict[str, Any] = dict(
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

    metadata_sd = get_metadata_state_dict(sharded_model, sharded_state_dict=True, device=DeviceGPU())
    assert 'sharded_state_dict' in metadata_sd
    assert metadata_sd['sharded_state_dict'] == True
    assert metadata_sd['model_name'] == expected_model_name

    if model_type == 'hf':
        assert 'huggingface' in metadata_sd
        assert 'model' in metadata_sd['huggingface']
        assert 'tokenizer' in metadata_sd['huggingface']
        assert 'model_name' in metadata_sd

    assert 'dist_backend' in metadata_sd
    assert metadata_sd['dist_backend'] == 'nccl'
