# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, OffloadWrapper
from torch.utils.data import DataLoader, Dataset

from composer.models import ComposerClassifier, ComposerModel
from composer.trainer.trainer import Trainer, _fsdp_reshard_and_cleanup
from composer.utils import dist
from composer.utils.parallelism import FSDPConfig
from tests.common import (
    EmbeddedWeightTiedModel,
    RandomClassificationDataset,
    SimpleModel,
    SimpleWeightTiedModel,
    world_size,
)

_INIT_DEVICES = ['cpu', 'meta', 'mixed', 'cuda']
_MIXED_PRECISION_TYPES = ['FULL', 'DEFAULT', 'PURE']


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, EmbeddedWeightTiedModel])
@pytest.mark.parametrize('mixed_precision', _MIXED_PRECISION_TYPES)
@pytest.mark.parametrize('device', _INIT_DEVICES)
@pytest.mark.parametrize('reentrant', [True, False])
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:The passed in model appears to have tied weights.*:UserWarning')
def test_fsdp_device_initialization(
    model: ComposerClassifier,
    mixed_precision: str,
    reentrant: bool,
    world_size: int,
    device: str,
):
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

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={
            'fsdp': {
                'activation_checkpointing_reentrant': reentrant,
                'mixed_precision': mixed_precision,
                'sync_module_states': True if device == 'mixed' else False,
            },
        },
        max_duration='3ba',
    )

    trainer.fit()
    if isinstance(model, SimpleWeightTiedModel):
        with trainer.state.model.module.summon_full_params(trainer.state.model.module):  # type: ignore
            fc1 = model.module[0].net[0]  # type: ignore
            fc2 = model.module[0].net[-1]  # type: ignore
            weight_1 = fc1.weight
            weight_2 = fc2.weight
            assert (id(weight_1) == id(weight_2))
            assert (torch.equal(weight_1, weight_2))

    if isinstance(model, EmbeddedWeightTiedModel):
        with trainer.state.model.module.summon_full_params(trainer.state.model.module):  # type: ignore
            weight_1 = model.module[0].net[0].weight  # type: ignore
            weight_2 = model.module[1].net[0].weight  # type: ignore
            assert (id(weight_1) == id(weight_2))
            assert (torch.equal(weight_1, weight_2))


@pytest.mark.parametrize(
    'model,expected_param_inits',
    [
        (SimpleModel, 2),  # One call for each of the Linear layers
        (EmbeddedWeightTiedModel, 3),  # Two calls for each of the SimpleMLP modules, minus one for weight tying
    ],
)
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
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

    Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={
            'fsdp': {
                'mixed_precision': 'PURE',
                'sharding_strategy': 'SHARD_GRAD_OP',
                'sync_module_states': True if device == 'mixed' else False,
            },
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

    Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={'fsdp': {
            'mixed_precision': mixed_precision,
            'sharding_strategy': 'SHARD_GRAD_OP',
        }},
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

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={
            'fsdp': {
                'forward_prefetch_limit': forward_prefetch_limit,
                'backward_prefetch_limit': backward_prefetch_limit,
            },
        },
        max_duration='3ba',
    )

    trainer.fit()


class SimpleDatasetForAuto(Dataset):

    def __init__(self, size: int = 256, feature_size: int = 1, num_classes: int = 2):
        self.size = size
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, self.feature_size)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,), dtype=torch.long)
        return self.x[index]


class SimpleMLPForTestingOOM(ComposerModel):

    def __init__(self, num_features: int = 128, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc2 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc3 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.rank = dist.get_global_rank()
        self.iter = 0

    def forward(self, x):
        x = self.fc1(x)
        if self.rank == 0 and x.shape[0] >= 64:
            raise RuntimeError('CUDA out of memory')
        x = self.fc2(x)
        x = self.fc3(x)
        self.iter += 1
        return x

    def loss(self, outputs, batch):
        return torch.sum(outputs)


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:`device_train_microbatch_size='auto'` may potentially fail with unexpected.*")
@pytest.mark.filterwarnings('ignore:CUDA out of memory*')
@world_size(2)
def test_automicrobatching_fsdp(world_size: int):
    model = SimpleMLPForTestingOOM()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    dataset = SimpleDatasetForAuto(size=256, feature_size=128)
    train_dataloader = DataLoader(dataset, batch_size=64, sampler=dist.get_sampler(dataset))
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        parallelism_config={
            'fsdp': {
                'forward_prefetch_limit': 1,
                'backward_prefetch_limit': 1,
            },
        },
        max_duration='1ba',
        device='gpu',
        device_train_microbatch_size='auto',
        dist_timeout=20,
    )
    trainer.fit()


class SimpleMLPForTestingHooks(ComposerModel):

    def __init__(self, num_features: int = 128, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc2 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc3 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.rank = dist.get_global_rank()
        self.iter = 0

    def forward(self, x):
        x = self.fc1(x)
        if self.iter == 3 and x.shape[0] >= 64:
            raise RuntimeError('CUDA out of memory')
        x = self.fc2(x)
        x = self.fc3(x)
        self.iter += 1
        return x

    def loss(self, outputs, batch):
        return torch.sum(outputs)


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:`device_train_microbatch_size='auto'` may potentially fail with unexpected.*")
@pytest.mark.filterwarnings('ignore:CUDA out of memory*')
@world_size(2)
def test_fsdp_automicrobatching_sync_hooks(world_size: int):
    model = SimpleMLPForTestingHooks()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    dataset = SimpleDatasetForAuto(size=256, feature_size=128)
    train_dataloader = DataLoader(dataset, batch_size=64, sampler=dist.get_sampler(dataset))

    with patch('composer.trainer.trainer._readd_fsdp_sync_hooks') as mock_readd_hooks:
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            parallelism_config={
                'fsdp': {
                    'forward_prefetch_limit': 1,
                    'backward_prefetch_limit': 1,
                },
            },
            max_duration='4ba',
            device='gpu',
            device_train_microbatch_size='auto',
            dist_timeout=20,
        )
        trainer.fit()

        # OOM occurs during the 4th batch, so check that sync hooks were readded at the end
        mock_readd_hooks.assert_called_once()


@pytest.mark.gpu
@world_size(2)
def test_fsdp_subset_of_params_in_opt(world_size: int):
    model = SimpleModel()
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.fc1.parameters(), lr=0.01)
    unwrapped_optimizer = copy.deepcopy(optimizer)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        parallelism_config={
            'fsdp': {
                'use_orig_params': True,
            },
        },
        max_duration='3ba',
    )

    # Validating that the model is a subclass of torch.nn.Module and has a valid callable for `summon_full_params`
    assert isinstance(trainer.state.model.module, torch.nn.Module)
    assert hasattr(trainer.state.model.module, 'summon_full_params')
    assert callable(trainer.state.model.module.summon_full_params)

    with trainer.state.model.module.summon_full_params(trainer.state.model.module):
        nb_parameters_before_fsdp = len(unwrapped_optimizer.param_groups[0]['params'])
        nb_parameters_after_fsdp = len(trainer.state.optimizers[0].param_groups[0]['params'])

        assert nb_parameters_before_fsdp == nb_parameters_after_fsdp


@pytest.mark.gpu
@world_size(2)
def test_fsdp_subset_of_params_in_opt_without_orig_params(world_size: int):
    model = SimpleModel()
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.fc1.parameters(), lr=0.01)

    expected_error = 'Passing in a subset of model parameters to the optimizer is not supported with use_orig_params=False.'

    with pytest.raises(ValueError, match=expected_error):
        _ = Trainer(
            model=model,
            optimizers=optimizer,
            train_dataloader=dataloader,
            parallelism_config={
                'fsdp': {
                    'use_orig_params': False,
                },
            },
            max_duration='3ba',
        )


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

    parallelism_config = {
        'fsdp': {
            'activation_checkpointing': activation_checkpointing,
            'activation_checkpointing_reentrant': False,
            'activation_cpu_offload': activation_cpu_offload,
        },
    }

    model.fc1._activation_checkpointing = True  # pyright: ignore[reportGeneralTypeIssues]

    trainer = Trainer(
        model=model,
        device='gpu',
        parallelism_config=parallelism_config,
    )

    assert trainer.state.fsdp_enabled

    assert isinstance(trainer.state.model.fc1, torch.nn.Module)

    if activation_checkpointing and activation_cpu_offload:
        assert isinstance(trainer.state.model.fc1._fsdp_wrapped_module, OffloadWrapper)
        assert isinstance(
            trainer.state.model.fc1._fsdp_wrapped_module._checkpoint_wrapped_module,
            CheckpointWrapper,
        )
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
        parallelism_config={'fsdp': {}},
        max_duration='3ba',
    )
    fsdp_model = trainer.state.model

    x = torch.rand([2, 128])
    output = fsdp_model(x)
    with pytest.raises(Exception):
        # Backward triggers the fake OOM exception,
        # which prevents fsdp reshard and cleanup
        torch.sum(output).backward()

    assert isinstance(fsdp_model.fc2, torch.nn.Module)

    fc2_flat_param = fsdp_model.fc2._flat_param

    # Without cleanup, model.fc2.flat_params is still in unshard state
    # the full param is not freed
    assert fc2_flat_param.data_ptr() != fc2_flat_param._local_shard.data_ptr()  # type: ignore
    assert fc2_flat_param._full_param_padded.numel() > 0  # type: ignore

    _fsdp_reshard_and_cleanup(fsdp_model)
    assert fc2_flat_param.data_ptr() == fc2_flat_param._local_shard.data_ptr()  # type: ignore
    assert fc2_flat_param._full_param_padded._typed_storage()._size() == 0  # type: ignore


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
        parallelism_config={'fsdp': {}},
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
        parallelism_config={'fsdp': {}},
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


@pytest.mark.gpu
@world_size(2)
def test_fsdp_device_mesh(world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

    # Expect error via pytest
    with pytest.raises(ValueError, match='Directly specifying device mesh for FSDP was deprecated*'):
        Trainer(
            model=model,
            parallelism_config={'fsdp': {
                'device_mesh': [2],
            }},
            max_duration='3ba',
        )


@pytest.mark.parametrize('error_key', ['device_mesh', '_device_mesh'])
def test_fsdp_config_device_mesh_error(error_key: str):
    # Passing device mesh directly to FSDPConfig should raise an error
    with pytest.raises(ValueError, match='Directly specifying device mesh for FSDP was deprecated*'):
        cfg_dict = {
            error_key: [2],
        }
        FSDPConfig(**cfg_dict)


@pytest.mark.gpu
@world_size(2)
def test_fsdp_shard(world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

    Trainer(
        model=model,
        parallelism_config={'fsdp': {
            'data_parallel_shard_degree': 2,
        }},
        max_duration='3ba',
    )


@pytest.mark.gpu
@world_size(2)
def test_fsdp_invalid_config_throws_error(world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

    expected_error = 'activation_cpu_offload=True is not supported with use_orig_params=False.'

    with pytest.raises(ValueError, match=expected_error):
        _ = Trainer(
            model=model,
            parallelism_config={
                'fsdp': {
                    'use_orig_params': False,
                    'activation_cpu_offload': True,
                },
            },
            max_duration='3ba',
        )


@pytest.mark.gpu
@world_size(2)
def test_fsdp_shard_and_replicate(world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]
    model.fc2._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

    Trainer(
        model=model,
        parallelism_config={
            'fsdp': {
                'data_parallel_shard_degree': 2,
                'data_parallel_replicate_degree': 1,
                'sharding_strategy': 'HYBRID_SHARD',
            },
        },
        max_duration='3ba',
    )
