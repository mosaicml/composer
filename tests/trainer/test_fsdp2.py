# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
import torch
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist, load_checkpoint
from composer.utils.parallelism import FSDP2Config, FSDPConfig, ParallelismConfig
from tests.common import (
    PartialWeightTiedModel,
    RandomClassificationDataset,
    SimpleComposerMLP,
    SimpleWeightTiedModel,
    world_size,
)
from tests.trainer.fsdp2_context import fsdp2_context, prepare_fully_shard


@fsdp2_context
def test_fsdp2_config():
    """Test that FSDP2Config read-only properties work as expected."""
    # Create a config instance
    config = FSDP2Config()

    # Test reading properties (should succeed)
    assert config.auto_wrap is False
    assert config.load_monolith_rank0_only is False
    assert config.sync_module_states is False
    assert config.activation_cpu_offload is False
    assert config.data_parallel_shard_degree == -1
    assert config.data_parallel_replicate_degree is None
    assert config.state_dict_type == 'sharded'
    assert config.use_orig_params is True

    # Test setting properties (should fail)
    read_only_props = [
        ('auto_wrap', False),
        ('load_monolith_rank0_only', True),
        ('sync_module_states', True),
        ('activation_cpu_offload', True),
        ('data_parallel_shard_degree', 2),
        ('data_parallel_replicate_degree', 2),
        ('state_dict_type', 'full'),
        ('use_orig_params', False),
    ]

    for prop, value in read_only_props:
        with pytest.raises(AttributeError):
            setattr(config, prop, value)

    # Test that core properties can be set
    config.device_mesh = None
    config.reshard_after_forward = False
    assert config.device_mesh is None
    assert config.reshard_after_forward is False


_INIT_DEVICES = ['cuda', 'meta']


def create_trainer_with_model(
    model: ComposerClassifier,
    num_classes: int = 10,
    max_duration: str = '10ep',
    use_fsdp2: bool = True,
) -> Trainer:
    """Helper function to create a Trainer with a model, dataloader, and FSDP2 configuration."""
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))

    parallelism_config = ParallelismConfig()
    if use_fsdp2:
        # Trainer is not calling prepare_fully_shard yet, so we need to do it manually
        fsdp2_config = FSDP2Config()
        # NOTE we can only apply FSDP2 to ComposerClassifier's module field until we support auto_wrap
        prepare_fully_shard(model=model.module, fsdp2_config=fsdp2_config)
        # NOTE module to_empty should only happen after the model is fully sharded and parameters are coverted to Dtensor
        # otherwise to_empty breaks weight tying
        # TODO (FSDP2) we should guardrail this in prepare_fully_shard
        model.to_empty(device='cuda')
        param_init_fn = getattr(model, 'param_init_fn', None)
        if param_init_fn is not None:
            for module in model.modules():
                param_init_fn(module)
        parallelism_config.fsdp2 = fsdp2_config
    else:
        parallelism_config.fsdp = FSDPConfig(state_dict_type='sharded')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        max_duration=max_duration,
        parallelism_config=parallelism_config,
    )
    return trainer


@pytest.mark.parametrize('model_class', [SimpleWeightTiedModel, PartialWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@fsdp2_context
def test_fsdp2_initialization_with_tied_params(
    model_class: type,
    device: str,
    world_size: int,
):
    """test FSDP2 initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied.
    """
    model = model_class(num_features=10, device=device)
    trainer = create_trainer_with_model(model=model,)

    # Initialization checks
    model = trainer.state.model
    assert isinstance(
        model,
        SimpleWeightTiedModel | PartialWeightTiedModel,
    ), f'Expected model to be SimpleWeightTiedModel or PartialWeightTiedModel, got {type(model)}'
    assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
    assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'
    assert len(model.mlp._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the mlp module'
    assert len(model.mlp.fc1._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc1 module'
    assert len(model.mlp.fc2._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc2 module'
    assert len(model.module._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the root module'
    if isinstance(model, PartialWeightTiedModel):
        assert len(model.fc3._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the fc3 module'
    assert model.mlp.fc1.weight.size(0) == model.mlp.fc2.weight.to_local(
    ).size(0) * world_size, 'Expect global weight size to be equal to local weight size * world_size on dim 0'

    trainer.fit()

    # Check that the weights are correctly tied
    weight_1 = model.mlp.fc1.weight.full_tensor()
    weight_2 = model.mlp.fc2.weight.full_tensor()
    assert (model.mlp.fc1.weight is model.mlp.fc2.weight)
    assert (torch.equal(weight_1, weight_2))


@pytest.mark.parametrize('model_class', [SimpleWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@fsdp2_context
def test_fsdp2_checkpointing(
    model_class: type,
    device: str,
    world_size: int,
    tmp_path: pathlib.Path,
):
    """Test FSDP2 checkpointing and weight tying after loading."""
    model = model_class(num_features=10, device=device)
    trainer = create_trainer_with_model(model=model,)

    # Checkpointing and reloading
    model = trainer.state.model
    assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
    assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
    assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'
    checkpoint_path = [tmp_path / 'dummy.pt']
    # Broadcast the path from rank 0 to all other ranks
    dist.broadcast_object_list(checkpoint_path, src=0)
    ckpt_path = trainer.save_checkpoint(str(checkpoint_path[0]), weights_only=True)
    assert isinstance(ckpt_path, str)

    # cache previous weights for comparison
    weight_1_local = model.mlp.fc1.weight.to_local()
    weight_2_local = model.mlp.fc2.weight.to_local()

    # reinitialize the trainer
    new_model = model_class(num_features=10, device=device)
    trainer = create_trainer_with_model(model=new_model,)
    load_checkpoint(str(pathlib.Path(ckpt_path).parent), trainer.state, trainer.logger, load_weights_only=True)

    model = trainer.state.model
    assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
    assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
    assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'
    # Check that the weights are still tied after loading and that the local weights are the same
    assert torch.equal(weight_1_local, model.mlp.fc1.weight.to_local())
    assert torch.equal(weight_2_local, model.mlp.fc2.weight.to_local())
    assert model.mlp.fc1.weight is model.mlp.fc2.weight


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
def test_fsdp2_load_from_fsdp1(
    world_size: int,
    tmp_path: pathlib.Path,
):
    """Test FSDP2 can load from FSDP1 checkpoint"""
    NUM_FEATURES = 10
    NUM_CLASSES = 2
    model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
    trainer = create_trainer_with_model(
        model=model,
        num_classes=NUM_CLASSES,
        use_fsdp2=False,
    )

    # Checkpointing
    model = trainer.state.model
    checkpoint_path = [tmp_path / 'dummy.pt']
    # Broadcast the path from rank 0 to all other ranks
    dist.broadcast_object_list(checkpoint_path, src=0)
    ckpt_path = trainer.save_checkpoint(str(checkpoint_path[0]), weights_only=True)
    assert isinstance(ckpt_path, str)

    # cache previous weights for comparison
    with model.module.summon_full_params(model.module):  # type: ignore
        fsdp1_param = [param.clone() for param in model.parameters()]

    # reinitialize the trainer
    model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
    trainer = create_trainer_with_model(
        model=model,
        num_classes=NUM_CLASSES,
        use_fsdp2=True,
    )
    load_checkpoint(str(pathlib.Path(ckpt_path).parent), trainer.state, trainer.logger, load_weights_only=True)
    for (name, param), fsdp1_param in zip(trainer.state.model.named_parameters(recurse=True), fsdp1_param):
        assert isinstance(param, DTensor), f'{name} should be a DTensor'
        assert torch.equal(
            fsdp1_param,
            param.full_tensor(),
        ), f'Weights: {name} should be equal after loading, however one is {fsdp1_param} and the other is {param.full_tensor()}'
