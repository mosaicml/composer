# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Optional

import pytest
import torch
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist, load_checkpoint
from composer.utils.parallelism import FSDP2Config, FSDPConfig, ParallelismConfig
from tests.common import (
    OOMComposerClassifier,
    PartialWeightTiedModel,
    RandomClassificationDataset,
    SimpleComposerMLP,
    SimpleWeightTiedModel,
    world_size,
)
from tests.trainer.fsdp2_context import (
    fsdp2_context,
    parallelize_model,
)


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
    size: int = 2,
    max_duration: str = '10ep',
    use_fsdp2: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    activation_checkpointing: bool = False,
    activation_cpu_offload: bool = False,
    auto_microbatching: bool = False,
) -> Trainer:
    """Helper function to create a Trainer with a model, dataloader, and FSDP2 configuration."""
    dataset = RandomClassificationDataset(shape=(num_classes,), size=size, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=size // 2) # use 2 batches per epoch
    hook_handles = []

    parallelism_config = ParallelismConfig()
    if use_fsdp2:
        # Trainer is not calling parallelize_model yet, so we need to do it manually
        fsdp2_config = FSDP2Config(
            activation_checkpointing=activation_checkpointing,
            activation_cpu_offload=activation_cpu_offload,
        )

        # NOTE we can only apply FSDP2 to ComposerClassifier's module field until we support auto_wrap
        hook_handles, _ = parallelize_model(model=model.module, config=fsdp2_config, optimizer=optimizer, auto_microbatching=auto_microbatching)
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
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        max_duration=max_duration,
        parallelism_config=parallelism_config,
        device_train_microbatch_size="auto" if auto_microbatching else None,
    )
    return trainer, hook_handles


# Base tests


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
    model.add_fsdp_wrap_attribute_to_children()
    trainer, _ = create_trainer_with_model(model=model,)

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


# Testing checkpointing and weight tying after loading


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
    model.add_fsdp_wrap_attribute_to_children()
    trainer, _ = create_trainer_with_model(model=model,)

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
    new_model.add_fsdp_wrap_attribute_to_children()
    trainer, _ = create_trainer_with_model(model=new_model,)
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
    model.add_fsdp_wrap_attribute_to_children()
    trainer, _ = create_trainer_with_model(
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
        # need clone since after context exit param be flat again
        fsdp1_param = [param.clone() for param in model.parameters()]

    # reinitialize the trainer
    model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
    model.add_fsdp_wrap_attribute_to_children()
    trainer, _ = create_trainer_with_model(
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


# Testing optimizer handling


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.parametrize('case', ['all_params_one_group', 'subset_one_group', 'multiple_groups'])
@pytest.mark.parametrize('device', _INIT_DEVICES)
def test_fsdp2_optimizer_handling(
    world_size: int,
    case: str,
    device: str,
):
    """Test FSDP2 correctly updates optimizer state for various configurations."""
    del world_size

    NUM_FEATURES = 10
    NUM_CLASSES = 10
    model = PartialWeightTiedModel(num_features=NUM_FEATURES, device=device)

    all_params_list = list(model.parameters())
    fc1_params_list = list(model.mlp.fc1.parameters())
    fc3_params_list = list(model.fc3.parameters())

    if case == 'all_params_one_group':
        optimizer_input = [{'params': all_params_list, 'lr': 0.01}]
    elif case == 'subset_one_group':
        optimizer_input = [{'params': fc1_params_list, 'lr': 0.02}]  # Same as fc2_params_list (since tied weights)
    elif case == 'multiple_groups':
        optimizer_input = [
            {
                'params': fc1_params_list,
                'lr': 0.01,
            },  # Same as fc2_params_list (since tied weights)
            {
                'params': fc3_params_list,
                'lr': 0.02,
            },
        ]
    else:
        raise ValueError(f'Invalid case: {case}')

    optimizer = torch.optim.Adam(optimizer_input)
    trainer, _ = create_trainer_with_model(model=model, num_classes=NUM_CLASSES, use_fsdp2=True, optimizer=optimizer)

    def validate_optimizer_state(current_optimizer: torch.optim.Optimizer, stage: str):
        assert len(current_optimizer.param_groups) == len(optimizer_input), \
            f'[{case}/{stage}] Group count mismatch. Expected {len(optimizer_input)}, Got {len(current_optimizer.param_groups)}'
        for i, group in enumerate(current_optimizer.param_groups):
            opt_params = group['params']
            # Check that the number of parameters in the optimizer group matches the number of parameters in the input
            assert len(opt_params) == len(optimizer_input[i]['params']), \
                f"[{case}/{stage}] Group {i}: Param count mismatch. Expected {len(optimizer_input[i]['params'])}, Got {len(opt_params)}"

            # Check that all parameters are DTensor
            assert all(isinstance(p, DTensor) for p in opt_params), \
                f'[{case}/{stage}] Group {i}: Not all parameters are DTensors'

            # Check that all keys match between input and current groups
            input_keys = set(optimizer_input[i].keys())
            group_keys = set(group.keys())
            assert input_keys == group_keys, \
                f'[{case}/{stage}] Group {i}: Key mismatch. Expected {input_keys}, Got {group_keys}'

            # Check values for all keys
            for key in input_keys:
                if key != 'params':
                    assert group[key] == optimizer_input[i][key], \
                        f'[{case}/{stage}] Group {i}: {key} mismatch. Expected {optimizer_input[i][key]}, Got {group[key]}'

    # Validate optimizer state after sharding and before training
    validate_optimizer_state(optimizer, stage='after_fully_shard')

    trainer.fit()

    # Validate optimizer state after training
    validate_optimizer_state(optimizer, stage='after_fit')


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
def test_fsdp2_optimizer_raises_error_when_optimizer_modules_dont_match(
    world_size: int,
):
    """Test FSDP2 raises an error when the optimizer modules don't match the model modules."""
    del world_size

    NUM_FEATURES = 10
    NUM_CLASSES = 10
    model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
    other_model = SimpleWeightTiedModel(num_features=NUM_FEATURES, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with pytest.raises(ValueError) as e:
        create_trainer_with_model(model=other_model, num_classes=NUM_CLASSES, use_fsdp2=True, optimizer=optimizer)
    # Check that error message uses the correct prefix implying optimizer difference
    # We check with `optimizer.param_id.` (with the period) since `optimizer.param_id` exists
    # by default in the error message's legend
    assert 'optimizer.param_id.' in str(e.value)


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.filterwarnings("ignore:`device_train_microbatch_size='auto'` may potentially fail with unexpected.*")
@pytest.mark.parametrize('use_alternate,num_layers,expected_num_hooks', [
    (False, 3, 3 * 2 + 1), # 3 children modules wrapped * 2 hook handles per module + 1 hook handle for the root module
    (True, 3, 2 * 2 + 2), # 2 children modules wrapped * 2 hook handles per module + 1 hook handles for the root module + 1 hook handle for last child module
])
def test_fsdp2_handles_cuda_failures(world_size: int, use_alternate: bool, num_layers: int, expected_num_hooks: int):
    """Test FSDP2 handles CUDA OOM failures."""
    del world_size

    num_classes = 10

    # Create a model with 3 layers, each wrapped in FSDP (so we should have 3 * 2 = 6 OOM hooks)
    model = OOMComposerClassifier(num_layers, num_classes, device='cuda')

    # Wrap the module as we expect
    model.module._fsdp_wrap = False
    for i, child in enumerate(model.module.children()):
        if use_alternate:
            child._fsdp_wrap = True if i % 2 == 0 else False  # type: ignore
        else:
            child._fsdp_wrap = True  # type: ignore

    # Assert that the number of hooks returned is correct
    _, hook_handles = create_trainer_with_model(model=model, num_classes=num_classes, use_fsdp2=True, auto_microbatching=True)
    assert len(hook_handles) == expected_num_hooks, f'Expected {expected_num_hooks} OOM hooks, but got {len(hook_handles)}'

    # Assert that all hook handles are RemovableHandle
    for hook_handle in hook_handles:
        assert isinstance(hook_handle, torch.utils.hooks.RemovableHandle), f'Expected RemovableHandle, but got {type(hook_handle)}'

    # Assert number of hooks on each module
    # Note: reshard_after_forward doesn't change the number of backward_hooks, it just changes the existing hooks do so the numbers
    # below are the same for both reshard_after_forward = True and False.
    for i, child in enumerate(model.module.children()):
        if use_alternate and i % 2 == 1:
            # This is the not FSDP wrapped module
            # We register one backward hook and no forward hooks. There are no FSDP hooks on this module as well.
            assert len(child._forward_pre_hooks) == 0, f'Expected 0 forward pre hooks on module {child}, but got {len(child._forward_pre_hooks)}'
            assert len(child._backward_pre_hooks) == 0, f'Expected 0 backward pre hooks on module {child}, but got {len(child._backward_pre_hooks)}'
            assert len(child._backward_hooks) == 1, f'Expected 1 backward hook on module {child}, but got {len(child._backward_hooks)}'
        else:
            # This is the FSDP wrapped module
            # We register one forward pre hook + the FSDP forward hook. We also register a backward pre hook
            assert len(child._forward_pre_hooks) == 2, f'Expected 2 forward pre hooks on module {child}, but got {len(child._forward_pre_hooks)}'
            assert len(child._backward_pre_hooks) == 1, f'Expected 1 backward pre hook on module {child}, but got {len(child._backward_pre_hooks)}'
            assert len(child._backward_hooks) == 0, f'Expected 0 backward hooks on module {child}, but got {len(child._backward_hooks)}'


@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:`device_train_microbatch_size='auto'` may potentially fail with unexpected.*")
@pytest.mark.filterwarnings('ignore:CUDA out of memory*')
@fsdp2_context
def test_fsdp2_auto_microbatching_handles_cuda_failures(world_size: int,):
    """Test FSDP2 auto-microbatching handles CUDA OOM failures."""
    del world_size

    # This will always fail (rank 1 will fail on all batch sizes)
    num_classes = 10
    model = OOMComposerClassifier(3, num_classes, device='cuda', always_fail=True)
    for child in model.module.children():
        child._fsdp_wrap = True
    trainer, _ = create_trainer_with_model(model=model, num_classes=num_classes, use_fsdp2=True, auto_microbatching=True, size=256, max_duration='1ba')
    with pytest.raises(RuntimeError, match='.*The train loop failed with an internal microbatch of size 1.*'):
        trainer.fit()

    # This will succeed as rank 1 will not fail if the microbatch size < 64
    model = OOMComposerClassifier(3, num_classes, device='cuda', always_fail=False)
    for child in model.module.children():
        child._fsdp_wrap = True
    trainer, _ = create_trainer_with_model(model=model, num_classes=num_classes, use_fsdp2=True, auto_microbatching=True, size=256, max_duration='1ba')
    trainer.fit()
