# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest.mock import MagicMock

import pytest
import torch
from packaging import version
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier, ComposerModel
from composer.trainer.trainer import Trainer, _fsdp_reshard_and_cleanup
from composer.utils import dist
from tests.common import (EmbeddedWeightTiedModel, RandomClassificationDataset, SimpleModel, SimpleWeightTiedModel,
                          world_size)

_INIT_DEVICES = ['cpu', 'meta', 'mixed', 'cuda']
_MIXED_PRECISION_TYPES = ['FULL', 'DEFAULT', 'PURE']


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, EmbeddedWeightTiedModel])
@pytest.mark.parametrize('mixed_precision', _MIXED_PRECISION_TYPES)
@pytest.mark.parametrize('device', _INIT_DEVICES)
@pytest.mark.parametrize('reentrant', [True, False])
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:The passed in model appears to have tied weights.*:UserWarning')
def test_fsdp_device_initialization(model: ComposerClassifier, mixed_precision: str, reentrant: bool, world_size: int,
                                    device: str):
    """test FSDP device initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied. This test also covers both 'cpu' and
    'meta' devices. This is because 'meta' will result in deferred initialization until FSDP is initialized

    """
    num_classes = 10

    resolved_device = device
    if device == 'mixed':
        if dist.get_local_rank() == 0:
            resolved_device = 'cpu'
        else:
            resolved_device = 'meta'
    model = model(num_features=num_classes, device=resolved_device)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'activation_checkpointing_reentrant': reentrant,
            'mixed_precision': mixed_precision,
            'sync_module_states': True if device == 'mixed' else False,
        },
        max_duration='3ba',
    )

    trainer.fit()
    if isinstance(model, SimpleWeightTiedModel):
        with trainer.state.model.module.summon_full_params(trainer.state.model.module):  # type: ignore
            weight_1 = model.mlp.fc1.weight
            weight_2 = model.mlp.fc2.weight
            assert (id(weight_1) == id(weight_2))
            assert (torch.equal(weight_1, weight_2))

    if isinstance(model, EmbeddedWeightTiedModel):
        with trainer.state.model.module.summon_full_params(trainer.state.model.module):  # type: ignore
            weight_1 = model.net1.fc1.weight
            weight_2 = model.net2.fc1.weight
            assert (id(weight_1) == id(weight_2))
            assert (torch.equal(weight_1, weight_2))


@pytest.mark.parametrize(
    'model,expected_param_inits',
    [
        (SimpleModel, 2),  # One call for each of the Linear layers
        (EmbeddedWeightTiedModel, 3),  # Two calls for each of the SimpleMLP modules, minus one for weight tying
    ])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.1.0'),
                    reason='This has only been fixed and tested starting with torch 2.1.0')
def test_fsdp_inits_params_once(model: ComposerClassifier, device: str, world_size: int, expected_param_inits: int):
    resolved_device = device
    if device == 'mixed':
        if dist.get_local_rank() == 0:
            resolved_device = 'cpu'
        else:
            resolved_device = 'meta'
    model = model(num_features=2, device=resolved_device)

    def dummy_param_init_fn(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.constant_(module.bias, 2)

    # Override the param_init_fn to be deterministic so we can test the init
    model.module.param_init_fn = dummy_param_init_fn  # pyright: ignore[reportGeneralTypeIssues]
    # Apply the initial initialization, because it will only be called later for parameters on meta device
    model.apply(model.module.param_init_fn)
    # Now wrap the param_init_fn with a MagicMock so we can count calls
    model.module.param_init_fn = MagicMock(wraps=model.module.param_init_fn)

    num_classes = 2
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'mixed_precision': 'PURE',
            'sharding_strategy': 'NO_SHARD',
            'sync_module_states': True if device == 'mixed' else False,
        },
        max_duration='3ba',
    )

    # We expect the param_init_fn to be called for each meta module, but not for modules already created on CPU
    if resolved_device == 'meta':
        assert model.module.param_init_fn.call_count == expected_param_inits
    else:
        assert model.module.param_init_fn.call_count == 0

    # Check that the parameters were initialized correctly
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert torch.all(module.weight == 1)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                assert torch.all(module.bias == 2)


@pytest.mark.parametrize('model', [SimpleModel])
@pytest.mark.parametrize('mixed_precision', _MIXED_PRECISION_TYPES)
@pytest.mark.gpu
@world_size(2)
def test_fsdp_meta_initialization_none(model: ComposerClassifier, mixed_precision: 'str', world_size: int):
    """
    This test is intended to test FSDP for meta initialization when there are attributes
    that are `None` and ensure we don't raise nasty UserWarnings.
    """
    num_classes = 2
    model = model(num_features=1, num_classes=num_classes, device='meta', bias=False)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'mixed_precision': mixed_precision,
            'sharding_strategy': 'NO_SHARD'
        },
        max_duration='3ba',
    )


@pytest.mark.parametrize('forward_prefetch_limit', [1, 2])
@pytest.mark.parametrize('backward_prefetch_limit', [1, 2])
@pytest.mark.gpu
@world_size(2)
def test_fsdp_prefetch_limit(forward_prefetch_limit: int, backward_prefetch_limit: int, world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'forward_prefetch_limit': forward_prefetch_limit,
            'backward_prefetch_limit': backward_prefetch_limit,
        },
        max_duration='3ba',
    )

    trainer.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.filterwarnings('ignore:Instantiating FSDP with custom process groups.*:UserWarning')
@pytest.mark.filterwarnings('ignore:Composer is instantiating custom process groups.*:UserWarning')
def test_fsdp_process_group(world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'process_group': 'mod1',  # all ranks
        },
        max_duration='3ba',
    )

    trainer.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.2.0'), reason='Device mesh requires Torch 2.2')
@pytest.mark.parametrize('sharding_strategy',
                         ['NO_SHARD', 'SHARD_GRAD_OP', 'FULL_SHARD', 'HYBRID_SHARD', '_HYBRID_SHARD_ZERO2'])
@pytest.mark.parametrize('device_mesh', [[2], [1, 2]])
def test_wrong_size_device_mesh_error(world_size: int, sharding_strategy: str, device_mesh: list[int]):
    context = contextlib.nullcontext()
    if sharding_strategy in ['NO_SHARD', 'SHARD_GRAD_OP', 'FULL_SHARD'] and len(device_mesh) != 1:
        context = pytest.raises(ValueError, match='.*requires a device mesh of size 1.*')
    if sharding_strategy in ['HYBRID_SHARD', '_HYBRID_SHARD_ZERO2'] and len(device_mesh) != 2:
        context = pytest.raises(ValueError, match='.*requires a device mesh of size 2.*')
    with context:
        Trainer(model=SimpleModel(), fsdp_config={
            'sharding_strategy': sharding_strategy,
            'device_mesh': device_mesh,
        })


class SimpleMLP(ComposerModel):

    def __init__(self, num_features: int = 128, device: str = 'cuda'):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc2 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def loss(self, outputs, batch):
        pass


@world_size(2)
@pytest.mark.gpu
@pytest.mark.parametrize('activation_checkpointing', [True, False])
@pytest.mark.parametrize('activation_cpu_offload', [True, False])
def test_fsdp_act_ckpt_offload(
    activation_checkpointing: bool,
    activation_cpu_offload: bool,
    world_size: int,
):
    model = SimpleMLP()

    fsdp_config = {
        'activation_checkpointing': activation_checkpointing,
        'activation_checkpointing_reentrant': False,
        'activation_cpu_offload': activation_cpu_offload,
    }

    model.fc1._activation_checkpointing = True  # pyright: ignore[reportGeneralTypeIssues]

    trainer = Trainer(
        model=model,
        device='gpu',
        fsdp_config=fsdp_config,
    )

    assert trainer.state.fsdp_enabled
    if version.parse(torch.__version__) > version.parse('2.1.0.dev'):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import OffloadWrapper

        if activation_checkpointing and activation_cpu_offload:
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, OffloadWrapper)
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module._checkpoint_wrapped_module,
                              CheckpointWrapper)
        elif activation_checkpointing:
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, CheckpointWrapper)
        elif activation_cpu_offload:
            assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, OffloadWrapper)
        else:
            assert not isinstance(trainer.state.model.fc1._fsdp_wrapped_module, CheckpointWrapper)


@pytest.mark.gpu
@world_size(2)
def test_fsdp_reshard_after_oom(world_size: int):
    model = SimpleMLP(num_features=128)
    model.relu._fsdp_wrap = False  # pyright: ignore[reportGeneralTypeIssues]

    def oom_hook(*args):
        raise RuntimeError('CUDA out of memory.')

    model.fc2.register_full_backward_hook(oom_hook)

    trainer = Trainer(
        model=model,
        fsdp_config={},
        max_duration='3ba',
    )
    fsdp_model = trainer.state.model

    x = torch.rand([2, 128])
    output = fsdp_model(x)
    with pytest.raises(Exception):
        # Backward triggers the fake OOM exception,
        # which prevents fsdp reshard and cleanup
        torch.sum(output).backward()

    fc2_flat_param = fsdp_model.fc2._flat_param

    # Without cleanup, model.fc2.flat_params is still in unshard state
    # the full param is not freed
    assert fc2_flat_param.data_ptr() != fc2_flat_param._local_shard.data_ptr()
    assert fc2_flat_param._full_param_padded.numel() > 0

    _fsdp_reshard_and_cleanup(fsdp_model)
    assert fc2_flat_param.data_ptr() == fc2_flat_param._local_shard.data_ptr()
    assert fc2_flat_param._full_param_padded._typed_storage()._size() == 0


@pytest.mark.gpu
@world_size(2)
def test_fsdp_same_state_after_oom_reshard(world_size: int):
    # Test numerical correctness after continuing to train with smaller batch size after OOM.
    model = SimpleMLP(num_features=2)
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.relu._fsdp_wrap = False  # pyright: ignore[reportGeneralTypeIssues]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        fsdp_config={},
        dist_timeout=20,
        optimizers=optimizer,
        seed=1,
    )
    fsdp_model = trainer.state.model

    state_dict = fsdp_model.state_dict()

    oom_model = SimpleMLP(num_features=2)
    oom_model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    oom_model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    oom_model.relu._fsdp_wrap = False  # pyright: ignore[reportGeneralTypeIssues]
    oom_model_optimizer = torch.optim.SGD(oom_model.parameters(), lr=0.1)

    def oom_hook(module, grad_input, grad_ouput):
        if grad_ouput[0].shape[0] >= 4:
            raise RuntimeError('CUDA out of memory.')

    oom_handle = oom_model.fc2.register_full_backward_hook(oom_hook)
    oom_trainer = Trainer(
        model=oom_model,
        fsdp_config={},
        dist_timeout=20,
        optimizers=oom_model_optimizer,
        seed=1,
    )

    fsdp_oom_model = oom_trainer.state.model
    fsdp_oom_model.load_state_dict(state_dict)

    x = torch.rand([4, 2])

    # Run fwd + bwd + optimizer on normal model
    output_0 = fsdp_model(x)
    torch.sum(output_0).backward()
    optimizer.step()

    # Run fwd + bwd + optimizer on OOM model
    output = fsdp_oom_model(x)
    with pytest.raises(Exception):
        torch.sum(output).backward()
    # Cleanup after OOM
    _fsdp_reshard_and_cleanup(fsdp_oom_model)
    oom_model_optimizer.zero_grad(set_to_none=True)

    oom_handle.remove()
    output = fsdp_oom_model(x)
    torch.sum(output).backward()
    oom_model_optimizer.step()

    # Run another fwd on both model and check
    # if output is the same
    output_1 = fsdp_model(x)
    output_2 = fsdp_oom_model(x)

    assert torch.equal(output_1, output_2)
