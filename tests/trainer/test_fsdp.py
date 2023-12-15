# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch
from packaging import version
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier, ComposerModel
from composer.trainer.trainer import Trainer
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
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='FSDP requires PyTorch 1.13 or higher')
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
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 2)

    # Override the param_init_fn to be deterministic so we can test the init
    model.module.param_init_fn = dummy_param_init_fn
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
            if module.bias is not None:
                assert torch.all(module.bias == 2)


@pytest.mark.parametrize('model', [SimpleModel])
@pytest.mark.parametrize('mixed_precision', _MIXED_PRECISION_TYPES)
@pytest.mark.gpu
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='FSDP requires PyTorch 1.13 or higher')
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
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='FSDP requires PyTorch 1.13 or higher')
def test_fsdp_prefetch_limit(forward_prefetch_limit: int, backward_prefetch_limit: int, world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True
    model.fc2._fsdp_wrap = True
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


class SimpleMLP(ComposerModel):

    def __init__(self, num_features: int = 128, device: str = 'cuda'):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_features, num_features, device=device, bias=False)
        self.fc2 = torch.nn.Linear(num_features, num_features, device=device, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.ReLU(x)
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

    model.fc1._activation_checkpointing = True

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
