# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from composer.algorithms import SWA
from composer.callbacks import SpeedMonitor
from composer.checkpoint import (
    get_metadata_state_dict,
    get_model_state_dict,
    get_optim_state_dict,
    get_resumption_state_dict,
)
from composer.core import State
from composer.devices import DeviceCPU, DeviceGPU
from composer.utils import dist, reproducibility
from tests.checkpoint.helpers import init_model_and_optimizer
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


@pytest.mark.gpu
@pytest.mark.parametrize('use_composer_model', [True, False])
def test_get_optim_state_dict_unsharded_model(use_composer_model: bool):
    model, optimizer = init_model_and_optimizer(use_composer_model=use_composer_model, take_step=True)
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
    model, optimizer = init_model_and_optimizer(use_composer_model=use_composer_model, take_step=True)
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

    model, optimizer = init_model_and_optimizer(
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

    model, optimizer = init_model_and_optimizer(
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
        'cuda_device_count',
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


@pytest.mark.filterwarnings('ignore:SWA has')
def test_get_resumption_state_dict():

    model, optimizer = init_model_and_optimizer(use_composer_model=True, take_step=True, device='cpu')

    rank_zero_seed = 10
    run_name = 'test_run'
    device = DeviceCPU()
    test_dataset_sd = {'foo': 0}
    dataloader = MagicMock(spec=DataLoader)
    dataloader.dataset = MagicMock()
    dataloader.dataset.state_dict = MagicMock(return_value=test_dataset_sd)
    swa = SWA()
    state = State(
        model=model,
        rank_zero_seed=rank_zero_seed,
        run_name=run_name,
        device=device,
        train_dataloader=dataloader,
        algorithms=[swa],
        callbacks=[SpeedMonitor(), SpeedMonitor()],
    )
    state.schedulers = StepLR(optimizer=optimizer, step_size=2)
    rsd = get_resumption_state_dict(state)

    assert rsd['rank_zero_seed'] == rank_zero_seed
    assert rsd['run_name'] == run_name
    assert 'timestamp' in rsd
    assert rsd['timestamp'] == {
        'iteration': 0,
        'epoch': 0,
        'batch': 0,
        'sample': 0,
        'token': 0,
        'epoch_in_iteration': 0,
        'batch_in_epoch': 0,
        'sample_in_epoch': 0,
        'token_in_epoch': 0,
        'total_wct': datetime.timedelta(0),
        'iteration_wct': datetime.timedelta(0),
        'epoch_wct': datetime.timedelta(0),
        'batch_wct': datetime.timedelta(0),
    }
    assert rsd['dataset_state'] == {'train': test_dataset_sd}
    dict(rsd['algorithms'])['SWA'].pop('repr')
    assert rsd['algorithms'] == [
        (
            'SWA',
            {
                'swa_model': None,
                'swa_completed': False,
                'swa_started': False,
                'swa_scheduler': None,
                'step_counter': 0,
            },
        ),
    ]
    assert rsd['callbacks'] == [('SpeedMonitor', {'total_eval_wct': 0.0}), ('SpeedMonitor', {'total_eval_wct': 0.0})]


@pytest.mark.gpu
def test_get_resumption_state_dict_gpu():
    if version.parse(torch.__version__) >= version.parse('2.3.0'):
        from torch.amp.grad_scaler import GradScaler
    else:
        from torch.cuda.amp.grad_scaler import GradScaler

    model, _ = init_model_and_optimizer(use_composer_model=True, take_step=False, device='cuda')

    rank_zero_seed = 10
    run_name = 'test_run'
    device = DeviceCPU()
    test_dataset_sd = {'test': 0}
    dataloader = MagicMock()
    dataloader.dataset = MagicMock()
    dataloader.dataset.state_dict = MagicMock(return_value=test_dataset_sd)
    state = State(
        model=model,
        rank_zero_seed=rank_zero_seed,
        run_name=run_name,
        device=device,
        scaler=GradScaler(),
    )
    rsd = get_resumption_state_dict(state)
    assert 'scaler' in rsd
    assert set(
        rsd['scaler'].keys(),
    ) == {'scale', 'growth_factor', 'backoff_factor', 'growth_interval', '_growth_tracker'}

    assert 'rng' in rsd
    deep_compare(rsd['rng'], reproducibility.get_rng_state())
