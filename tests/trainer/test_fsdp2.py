# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import Callable, Optional

import pytest
import torch
from torch.distributed._tensor import DTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
    OffloadWrapper,
)
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist, load_checkpoint
from composer.utils.parallelism import FSDP2Config, FSDPConfig, ParallelismConfig
from tests.common import (
    ComposerCounterModel,
    CountModule,
    PartialWeightTiedModel,
    RandomClassificationDataset,
    SimpleComposerMLP,
    SimpleWeightTiedModel,
    world_size,
)
from tests.trainer.fsdp2_context import (
    fsdp2_context,
    prepare_fully_shard,
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
    optimizer: Optional[torch.optim.Optimizer] = None,
    activation_checkpointing: bool = False,
    activation_cpu_offload: bool = False,
) -> Trainer:
    """Helper function to create a Trainer with a model, dataloader, and FSDP2 configuration."""
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))

    parallelism_config = ParallelismConfig()
    if use_fsdp2:
        # Trainer is not calling prepare_fully_shard yet, so we need to do it manually
        fsdp2_config = FSDP2Config(
            activation_checkpointing=activation_checkpointing,
            activation_cpu_offload=activation_cpu_offload,
        )

        # NOTE we can only apply FSDP2 to ComposerClassifier's module field until we support auto_wrap
        prepare_fully_shard(model=model.module, fsdp2_config=fsdp2_config, optimizer=optimizer)
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
    )
    return trainer


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
    new_model.add_fsdp_wrap_attribute_to_children()
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
    model.add_fsdp_wrap_attribute_to_children()
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
        # need clone since after context exit param be flat again
        fsdp1_param = [param.clone() for param in model.parameters()]

    # reinitialize the trainer
    model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
    model.add_fsdp_wrap_attribute_to_children()
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
    trainer = create_trainer_with_model(model=model, num_classes=NUM_CLASSES, use_fsdp2=True, optimizer=optimizer)

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
def test_fsdp2_optimizer_raises_error_when_optimizer_modules_dont_match(world_size: int,):
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


# Testing activation checkpointing


def validate_activation_wrapper(
    module: torch.nn.Module,
    is_activation_checkpoint_enabled: bool,
    is_activation_offload_enabled: bool,
    checkpoint_fn: Optional[Callable] = None,
) -> None:
    """
    Verify that activation checkpointing and offload wrappers exist where expected
    based on the `_activation_checkpointing` flag or `checkpoint_fn` and the
    provided boolean flags.

    Raises ValueError if validation fails, listing the offending module names.

    Args:
        module (torch.nn.Module): The root model module to inspect.
        is_activation_checkpointed (bool): Whether activation checkpointing wrappers (`ActivationWrapper`) are expected.
        is_activation_offloaded (bool): Whether activation offload wrappers (`OffloadWrapper`) are expected.
        checkpoint_fn (Optional[Callable]): An optional function to determine if a module should be checkpointed.
    """
    offenders: list[str] = []

    def check_module_needs_wrapping(inner_module: torch.nn.Module) -> bool:
        if checkpoint_fn is not None and checkpoint_fn(inner_module):
            return True
        return hasattr(inner_module, '_activation_checkpointing') and bool(inner_module._activation_checkpointing)

    def _walk(mod: torch.nn.Module, prefix: str) -> None:
        inner_mod = mod
        actual_is_offloaded = isinstance(mod, OffloadWrapper)
        actual_is_checkpointed = False

        if actual_is_offloaded:
            offload_inner = getattr(mod, _CHECKPOINT_WRAPPED_MODULE, None)
            if offload_inner is None:
                # We shouldn't ever see this, but it's good to have the check
                offenders.append(f'{prefix}: OffloadWrapper missing inner module')
                return

            actual_is_checkpointed = isinstance(offload_inner, ActivationWrapper)
            if actual_is_checkpointed:
                inner_mod = getattr(offload_inner, _CHECKPOINT_WRAPPED_MODULE, None)
                if inner_mod is None:
                    # We shouldn't ever see this, but it's good to have the check
                    offenders.append(f'{prefix}: ActivationWrapper missing inner module')
                    return
            else:
                inner_mod = offload_inner
        elif isinstance(mod, ActivationWrapper):
            actual_is_checkpointed = True
            inner_mod = getattr(mod, _CHECKPOINT_WRAPPED_MODULE, None)
            if inner_mod is None:
                # We shouldn't ever see this, but it's good to have the check
                offenders.append(f'{prefix}: ActivationWrapper missing inner module')
                return

        module_needs_wrapping = check_module_needs_wrapping(inner_mod)

        validation_failed = False
        error_reasons = []
        if module_needs_wrapping:
            if actual_is_checkpointed != is_activation_checkpoint_enabled:
                validation_failed = True
                error_reasons.append(f'Expected checkpointed={is_activation_checkpoint_enabled}, Got={actual_is_checkpointed}')
            if actual_is_offloaded != is_activation_offload_enabled:
                validation_failed = True
                error_reasons.append(f'Expected offloaded={is_activation_offload_enabled}, Got={actual_is_offloaded}')
        else:
            if actual_is_checkpointed:
                validation_failed = True
                error_reasons.append('Should not be checkpointed, but is')
            if actual_is_offloaded:
                validation_failed = True
                error_reasons.append('Should not be offloaded, but is')

        if validation_failed:
            offenders.append(f"{prefix or inner_mod.__class__.__name__}: {', '.join(error_reasons)}")

        # Recurse on the children of the *original* module structure
        for name, child in inner_mod.named_children():
            child_prefix = f'{prefix}.{name}' if prefix else name
            _walk(child, child_prefix)

    _walk(module, '')
    if len(offenders) > 0:
        raise ValueError(f'Activation wrapper validation failed. Offending modules: {offenders}')


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.parametrize('activation_checkpointing,expected_forward_count,activation_cpu_offload',
    [
        (True, 2, True),
        (True, 2, False),
        (False, 1, True),
        (False, 1, False),
    ],
)
def test_fsdp2_activation_checkpointing_attribute(
    world_size: int,
    activation_checkpointing: bool,
    expected_forward_count: int,
    activation_cpu_offload: bool,
):
    """Test FSDP2 activation checkpointing."""
    del world_size

    model = ComposerCounterModel(num_inputs=10, num_outputs=10, num_classes=10, device='cuda')
    if activation_checkpointing:
        model.module[0]._activation_checkpointing = True  # type: ignore

    # Train the model on one batch to make sure forward is called the expected number of times
    trainer = create_trainer_with_model(
        model=model,
        num_classes=10,
        use_fsdp2=True,
        activation_checkpointing=activation_checkpointing,
        activation_cpu_offload=activation_cpu_offload,
        max_duration='1ba',
    )
    print(trainer.state.model.module)
    # Validate that the activation checkpointing wrapper is applied correctly pre-training
    validate_activation_wrapper(trainer.state.model.module, activation_checkpointing, activation_cpu_offload)  # type: ignore
    trainer.fit()
    # validate that the activation checkpointing wrapper is applied correctly post-training
    validate_activation_wrapper(trainer.state.model.module, activation_checkpointing, activation_cpu_offload)  # type: ignore
    error_msg = 'forward hook called {actual_forward_count} times, but expected {expected_forward_count} times.'
    counter_module_0_call_count = model.module[0].call_count  # type: ignore
    counter_module_1_call_count = model.module[-1].call_count  # type: ignore
    assert counter_module_0_call_count == expected_forward_count, \
        error_msg.format(expected_forward_count=expected_forward_count, actual_forward_count=counter_module_0_call_count)
    assert counter_module_1_call_count == 1, 'Expected last module to be called once since it is not checkpointed'


@world_size(2)
@pytest.mark.gpu
@fsdp2_context
@pytest.mark.parametrize('activation_checkpointing,expected_forward_count,activation_cpu_offload',
    [
        (True, 2, True),
        (True, 2, False),
        (False, 1, True),
        (False, 1, False),
    ],
)
def test_fsdp2_activation_checkpointing_fn(
    world_size: int,
    activation_checkpointing: bool,
    expected_forward_count: int,
    activation_cpu_offload: bool,
):
    """Test FSDP2 activation checkpointing."""
    del world_size

    model = ComposerCounterModel(num_inputs=10, num_outputs=10, num_classes=10, device='cuda')
    activation_checkpointing_fn = None  # type: ignore

    # Checkpoint both CountModules
    if activation_checkpointing:
        def activation_checkpointing_fn(module: torch.nn.Module) -> bool:
            return isinstance(module, CountModule)
        model.module.activation_checkpointing_fn = activation_checkpointing_fn  # type: ignore

    # Train the model on one batch to make sure forward is called the expected number of times
    trainer = create_trainer_with_model(
        model=model,
        num_classes=10,
        use_fsdp2=True,
        activation_checkpointing=activation_checkpointing,
        activation_cpu_offload=activation_cpu_offload,
        max_duration='1ba',
    )
    # Validate that the activation checkpointing wrapper is applied correctly pre-training
    validate_activation_wrapper(trainer.state.model.module, activation_checkpointing, activation_cpu_offload, activation_checkpointing_fn)  # type: ignore
    trainer.fit()
    # validate that the activation checkpointing wrapper is applied correctly post-training
    validate_activation_wrapper(trainer.state.model.module, activation_checkpointing, activation_cpu_offload, activation_checkpointing_fn)  # type: ignore

    error_msg = 'forward hook called {actual_forward_count} times, but expected {expected_forward_count} times.'
    counter_module_0_call_count = model.module[0].call_count  # type: ignore
    counter_module_1_call_count = model.module[-1].call_count  # type: ignore
    assert counter_module_0_call_count == expected_forward_count, \
        error_msg.format(expected_forward_count=expected_forward_count, actual_forward_count=counter_module_0_call_count)
    assert counter_module_1_call_count == expected_forward_count, \
        error_msg.format(expected_forward_count=expected_forward_count, actual_forward_count=counter_module_1_call_count)
