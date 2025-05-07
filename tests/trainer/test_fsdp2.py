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


_INIT_DEVICES = ['meta']


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
        parallelism_config.fsdp2 = FSDP2Config(
            activation_checkpointing=activation_checkpointing,
            activation_cpu_offload=activation_cpu_offload,
        )
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


@pytest.mark.parametrize('model_class', [SimpleWeightTiedModel])
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
    # print(model)
    # print("--------------------------------")
    # for module in model.modules():
    #     print(module)
    # return
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
    # assert len(model.module._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the root module'
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


# @pytest.mark.parametrize('model_class', [SimpleWeightTiedModel])
# @pytest.mark.parametrize('device', _INIT_DEVICES)
# @world_size(2)
# @pytest.mark.gpu
# @fsdp2_context
# def test_fsdp2_checkpointing(
#     model_class: type,
#     device: str,
#     world_size: int,
#     tmp_path: pathlib.Path,
# ):
#     """Test FSDP2 checkpointing and weight tying after loading."""
#     model = model_class(num_features=10, device=device)
#     model.add_fsdp_wrap_attribute_to_children()
#     trainer = create_trainer_with_model(model=model,)

#     # Checkpointing and reloading
#     model = trainer.state.model
#     assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
#     assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
#     assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'
#     checkpoint_path = [tmp_path / 'dummy.pt']
#     # Broadcast the path from rank 0 to all other ranks
#     dist.broadcast_object_list(checkpoint_path, src=0)
#     ckpt_path = trainer.save_checkpoint(str(checkpoint_path[0]), weights_only=True)
#     assert isinstance(ckpt_path, str)

#     # cache previous weights for comparison
#     weight_1_local = model.mlp.fc1.weight.to_local()
#     weight_2_local = model.mlp.fc2.weight.to_local()

#     # reinitialize the trainer
#     new_model = model_class(num_features=10, device=device)
#     new_model.add_fsdp_wrap_attribute_to_children()
#     trainer = create_trainer_with_model(model=new_model,)
#     load_checkpoint(str(pathlib.Path(ckpt_path).parent), trainer.state, trainer.logger, load_weights_only=True)

#     model = trainer.state.model
#     assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
#     assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
#     assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'
#     # Check that the weights are still tied after loading and that the local weights are the same
#     assert torch.equal(weight_1_local, model.mlp.fc1.weight.to_local())
#     assert torch.equal(weight_2_local, model.mlp.fc2.weight.to_local())
#     assert model.mlp.fc1.weight is model.mlp.fc2.weight


# @world_size(2)
# @pytest.mark.gpu
# @fsdp2_context
# def test_fsdp2_load_from_fsdp1(
#     world_size: int,
#     tmp_path: pathlib.Path,
# ):
#     """Test FSDP2 can load from FSDP1 checkpoint"""
#     NUM_FEATURES = 10
#     NUM_CLASSES = 2
#     model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
#     model.add_fsdp_wrap_attribute_to_children()
#     trainer = create_trainer_with_model(
#         model=model,
#         num_classes=NUM_CLASSES,
#         use_fsdp2=False,
#     )

#     # Checkpointing
#     model = trainer.state.model
#     checkpoint_path = [tmp_path / 'dummy.pt']
#     # Broadcast the path from rank 0 to all other ranks
#     dist.broadcast_object_list(checkpoint_path, src=0)
#     ckpt_path = trainer.save_checkpoint(str(checkpoint_path[0]), weights_only=True)
#     assert isinstance(ckpt_path, str)

#     # cache previous weights for comparison
#     with model.module.summon_full_params(model.module):  # type: ignore
#         # need clone since after context exit param be flat again
#         fsdp1_param = [param.clone() for param in model.parameters()]

#     # reinitialize the trainer
#     model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
#     model.add_fsdp_wrap_attribute_to_children()
#     trainer = create_trainer_with_model(
#         model=model,
#         num_classes=NUM_CLASSES,
#         use_fsdp2=True,
#     )
#     load_checkpoint(str(pathlib.Path(ckpt_path).parent), trainer.state, trainer.logger, load_weights_only=True)
#     for (name, param), fsdp1_param in zip(trainer.state.model.named_parameters(recurse=True), fsdp1_param):
#         assert isinstance(param, DTensor), f'{name} should be a DTensor'
#         assert torch.equal(
#             fsdp1_param,
#             param.full_tensor(),
#         ), f'Weights: {name} should be equal after loading, however one is {fsdp1_param} and the other is {param.full_tensor()}'


# # Testing optimizer handling


# @world_size(2)
# @pytest.mark.gpu
# @fsdp2_context
# @pytest.mark.parametrize('case', ['all_params_one_group', 'subset_one_group', 'multiple_groups'])
# @pytest.mark.parametrize('device', _INIT_DEVICES)
# def test_fsdp2_optimizer_handling(
#     world_size: int,
#     case: str,
#     device: str,
# ):
#     """Test FSDP2 correctly updates optimizer state for various configurations."""
#     del world_size

#     NUM_FEATURES = 10
#     NUM_CLASSES = 10
#     model = PartialWeightTiedModel(num_features=NUM_FEATURES, device=device)

#     all_params_list = list(model.parameters())
#     fc1_params_list = list(model.mlp.fc1.parameters())
#     fc3_params_list = list(model.fc3.parameters())

#     if case == 'all_params_one_group':
#         optimizer_input = [{'params': all_params_list, 'lr': 0.01}]
#     elif case == 'subset_one_group':
#         optimizer_input = [{'params': fc1_params_list, 'lr': 0.02}]  # Same as fc2_params_list (since tied weights)
#     elif case == 'multiple_groups':
#         optimizer_input = [
#             {
#                 'params': fc1_params_list,
#                 'lr': 0.01,
#             },  # Same as fc2_params_list (since tied weights)
#             {
#                 'params': fc3_params_list,
#                 'lr': 0.02,
#             },
#         ]
#     else:
#         raise ValueError(f'Invalid case: {case}')

#     optimizer = torch.optim.Adam(optimizer_input)
#     trainer = create_trainer_with_model(model=model, num_classes=NUM_CLASSES, use_fsdp2=True, optimizer=optimizer)

#     def validate_optimizer_state(current_optimizer: torch.optim.Optimizer, stage: str):
#         assert len(current_optimizer.param_groups) == len(optimizer_input), \
#             f'[{case}/{stage}] Group count mismatch. Expected {len(optimizer_input)}, Got {len(current_optimizer.param_groups)}'
#         for i, group in enumerate(current_optimizer.param_groups):
#             opt_params = group['params']
#             # Check that the number of parameters in the optimizer group matches the number of parameters in the input
#             assert len(opt_params) == len(optimizer_input[i]['params']), \
#                 f"[{case}/{stage}] Group {i}: Param count mismatch. Expected {len(optimizer_input[i]['params'])}, Got {len(opt_params)}"

#             # Check that all parameters are DTensor
#             assert all(isinstance(p, DTensor) for p in opt_params), \
#                 f'[{case}/{stage}] Group {i}: Not all parameters are DTensors'

#             # Check that all keys match between input and current groups
#             input_keys = set(optimizer_input[i].keys())
#             group_keys = set(group.keys())
#             assert input_keys == group_keys, \
#                 f'[{case}/{stage}] Group {i}: Key mismatch. Expected {input_keys}, Got {group_keys}'

#             # Check values for all keys
#             for key in input_keys:
#                 if key != 'params':
#                     assert group[key] == optimizer_input[i][key], \
#                         f'[{case}/{stage}] Group {i}: {key} mismatch. Expected {optimizer_input[i][key]}, Got {group[key]}'

#     # Validate optimizer state after sharding and before training
#     validate_optimizer_state(optimizer, stage='after_fully_shard')

#     trainer.fit()

#     # Validate optimizer state after training
#     validate_optimizer_state(optimizer, stage='after_fit')


# @world_size(2)
# @pytest.mark.gpu
# @fsdp2_context
# def test_fsdp2_optimizer_raises_error_when_optimizer_modules_dont_match(world_size: int,):
#     """Test FSDP2 raises an error when the optimizer modules don't match the model modules."""
#     del world_size

#     NUM_FEATURES = 10
#     NUM_CLASSES = 10
#     model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
#     other_model = SimpleWeightTiedModel(num_features=NUM_FEATURES, device='cuda')
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     with pytest.raises(ValueError) as e:
#         create_trainer_with_model(model=other_model, num_classes=NUM_CLASSES, use_fsdp2=True, optimizer=optimizer)
#     # Check that error message uses the correct prefix implying optimizer difference
#     # We check with `optimizer.param_id.` (with the period) since `optimizer.param_id` exists
#     # by default in the error message's legend
#     assert 'optimizer.param_id.' in str(e.value)
