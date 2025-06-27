# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
from typing import Optional

import pytest
import torch
import torch.distributed.fsdp
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
from tests.trainer.test_fsdp_checkpoint import _assert_checkpoints_equivalent

_INIT_DEVICES = ['cuda', 'meta']


def create_trainer_with_model(
    model: ComposerClassifier,
    num_classes: int = 10,
    dataset_size: int = 2,
    batch_size: int = 1,
    max_duration: str = '10ep',
    use_fsdp2: bool = True,
    optimizer: Optional[torch.optim.Optimizer] = None,
    activation_checkpointing: bool = False,
    activation_cpu_offload: bool = False,
    auto_microbatching: bool = False,
    fsdp1_sync_module_states: bool = False,
    state_dict_type: str = 'sharded',
    load_monolith_rank0_only: bool = False,
    save_folder: Optional[str] = None,
    save_filename: str = 'ba{batch}-rank{rank}.pt',
    save_interval: str = '10ba',
    load_path: Optional[str] = None,
    precision: Optional[str] = None,
    mixed_precision: str = 'PURE',
) -> Trainer:
    """Helper function to create a Trainer with a model, dataloader, and FSDP2 configuration."""
    dataset = RandomClassificationDataset(shape=(num_classes,), size=dataset_size, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=batch_size)

    parallelism_config = ParallelismConfig()
    if use_fsdp2:
        parallelism_config.fsdp2 = FSDP2Config(
            activation_checkpointing=activation_checkpointing,
            activation_cpu_offload=activation_cpu_offload,
            state_dict_type=state_dict_type,
            load_monolith_rank0_only=load_monolith_rank0_only,
            mixed_precision=mixed_precision,
        )
    else:
        parallelism_config.fsdp = FSDPConfig(
            state_dict_type=state_dict_type,
            sync_module_states=fsdp1_sync_module_states,
        )
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        optimizers=optimizer,
        max_duration=max_duration,
        parallelism_config=parallelism_config,
        device_train_microbatch_size='auto' if auto_microbatching else None,
        save_folder=save_folder,
        save_filename=save_filename,
        save_interval=save_interval,
        load_path=load_path,
        precision=precision,
    )
    return trainer


# Base tests


@pytest.mark.parametrize('model_class', [SimpleWeightTiedModel, PartialWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
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
    trainer = create_trainer_with_model(
        model=model,
    )

    # Initialization checks
    model = trainer.state.model
    assert isinstance(
        model,
        SimpleWeightTiedModel | PartialWeightTiedModel,
    ), f'Expected model to be SimpleWeightTiedModel or PartialWeightTiedModel, got {type(model)}'

    mlp = model.module[0]  # type: ignore
    fc1 = mlp.net[0]  # type: ignore
    fc2 = mlp.net[-1]  # type: ignore

    assert isinstance(fc1.weight, DTensor), 'fc1.weight should be a DTensor'
    assert isinstance(fc2.weight, DTensor), 'fc2.weight should be a DTensor'
    assert len(mlp._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the mlp module'
    assert len(fc1._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc1 module'
    assert len(fc2._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc2 module'
    if isinstance(model, PartialWeightTiedModel):
        fc3 = model.module[-1]  # type: ignore
        assert len(fc3._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the fc3 module'
    assert fc1.weight.size(0) == fc2.weight.to_local().size(0) * world_size, \
        'Expect global weight size to be equal to local weight size * world_size on dim 0'

    trainer.fit()

    # Check that the weights are correctly tied
    weight_1 = fc1.weight.full_tensor()
    weight_2 = fc2.weight.full_tensor()
    assert (fc1.weight is fc2.weight)
    assert (torch.equal(weight_1, weight_2))


# Testing checkpointing and weight tying after loading


@pytest.mark.parametrize('model_class', [SimpleWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
def test_fsdp2_checkpointing(
    model_class: type,
    device: str,
    world_size: int,
    tmp_path: pathlib.Path,
):
    """Test FSDP2 checkpointing and weight tying after loading."""
    model = model_class(num_features=10, device=device)
    model.add_fsdp_wrap_attribute_to_children()
    trainer = create_trainer_with_model(
        model=model,
    )

    # Checkpointing and reloading
    model = trainer.state.model

    mlp = model.module[0]  # type: ignore
    fc1 = mlp.net[0]  # type: ignore
    fc2 = mlp.net[-1]  # type: ignore

    assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
    assert isinstance(fc1.weight, DTensor), 'fc1.weight should be a DTensor'
    assert isinstance(fc2.weight, DTensor), 'fc2.weight should be a DTensor'
    checkpoint_path = [tmp_path / 'dummy.pt']
    # Broadcast the path from rank 0 to all other ranks
    dist.broadcast_object_list(checkpoint_path, src=0)
    ckpt_path = trainer.save_checkpoint(str(checkpoint_path[0]), weights_only=True)
    assert isinstance(ckpt_path, str)

    # cache previous weights for comparison
    weight_1_local = fc1.weight.to_local()
    weight_2_local = fc2.weight.to_local()

    # reinitialize the trainer
    new_model = model_class(num_features=10, device=device)
    new_model.add_fsdp_wrap_attribute_to_children()
    trainer = create_trainer_with_model(
        model=new_model,
    )
    load_checkpoint(str(pathlib.Path(ckpt_path).parent), trainer.state, trainer.logger, load_weights_only=True)

    model = trainer.state.model
    assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
    assert isinstance(fc1.weight, DTensor), 'fc1.weight should be a DTensor'
    assert isinstance(fc2.weight, DTensor), 'fc2.weight should be a DTensor'
    # Check that the weights are still tied after loading and that the local weights are the same
    assert torch.equal(weight_1_local, fc1.weight.to_local())
    assert torch.equal(weight_2_local, fc2.weight.to_local())
    assert fc1.weight is fc2.weight


@world_size(2)
@pytest.mark.gpu
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
    model.add_fsdp_wrap_attribute_to_children()

    all_params_list = list(model.parameters())
    fc1_params_list = list(model.module[0].net[0].parameters())  # type: ignore
    fc3_params_list = list(model.module[-1].parameters())  # type: ignore

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
def test_fsdp2_optimizer_raises_error_when_optimizer_modules_dont_match(
    world_size: int,
):
    """Test FSDP2 raises an error when the optimizer modules don't match the model modules."""
    del world_size

    NUM_FEATURES = 10
    NUM_CLASSES = 10
    model = SimpleComposerMLP(num_features=NUM_FEATURES, device='cuda', num_classes=NUM_CLASSES)
    model.add_fsdp_wrap_attribute_to_children()
    other_model = SimpleWeightTiedModel(num_features=NUM_FEATURES, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    other_model.add_fsdp_wrap_attribute_to_children()
    with pytest.raises(ValueError) as e:
        create_trainer_with_model(model=other_model, num_classes=NUM_CLASSES, use_fsdp2=True, optimizer=optimizer)
    # Check that error message uses the correct prefix implying optimizer difference
    # We check with `optimizer.param_id.` (with the period) since `optimizer.param_id` exists
    # by default in the error message's legend
    assert 'optimizer.param_id.' in str(e.value)


@pytest.mark.gpu
@world_size(2)  # Using world_size(2) for consistency with other FSDP2 tests in this file although not needed
@pytest.mark.filterwarnings("ignore:`device_train_microbatch_size='auto'` may potentially fail with unexpected.*")
def test_fsdp2_auto_microbatching_raises_error(
    world_size: int,
):
    """Test FSDP2 raises an error when auto microbatching is used."""
    del world_size

    model = SimpleComposerMLP(num_features=10, device='cuda', num_classes=10)
    model.add_fsdp_wrap_attribute_to_children()
    with pytest.raises(ValueError) as e:
        create_trainer_with_model(model=model, num_classes=10, use_fsdp2=True, auto_microbatching=True)
    assert 'Auto microbatching is not supported outside of FSDP1' in str(e.value)


class TestFSDP2MixedInit:
    """Test class for FSDP2 mixed initialization scenarios."""

    @staticmethod
    def _set_deterministic_seed(seed: int):
        """Helper function to set seed consistently across all random number generators."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _create_model_with_mixed_init(model_class: type, num_features: int, device: str, seed: int = 42):
        """Helper function to create a model with mixed initialization (CPU for rank 0, meta for others)."""
        TestFSDP2MixedInit._set_deterministic_seed(seed)

        resolved_device = device if dist.get_local_rank() == 0 else 'meta'

        # set the bias to be True for SimpleComposerMLP
        # which is used for a later test
        kwargs = {}
        if model_class == SimpleComposerMLP:
            kwargs['add_bias'] = True

        model = model_class(num_features=num_features, device=resolved_device, **kwargs)
        model.add_fsdp_wrap_attribute_to_children()

        if dist.get_local_rank() == 0:
            model.apply(model.param_init_fn)

        return model

    @staticmethod
    def _train_model_and_extract_weights(model, use_fsdp2: bool):
        """Helper function to train a mixed init model and extract its weights."""
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        kwargs = {}

        # If we're using FSDP1, we need to manually set the
        # attribute to sync the module states during mixed init
        if not use_fsdp2:
            kwargs['fsdp1_sync_module_states'] = True
        trainer = create_trainer_with_model(
            model=model,
            max_duration=f'10ba',
            use_fsdp2=use_fsdp2,
            optimizer=optimizer,
            **kwargs,
        )

        trainer.fit()

        weights = {}
        if use_fsdp2:
            for name, param in trainer.state.model.named_parameters():
                weights[name] = param.full_tensor().data.clone()  # type: ignore
        else:
            with trainer.state.model.module.summon_full_params(trainer.state.model.module):  # type: ignore
                for name, param in trainer.state.model.named_parameters():
                    weights[name] = param.data.clone()

        return weights

    @staticmethod
    def _compare_weights(fsdp1_weights: dict, fsdp2_weights: dict, tolerance: float = 1e-4):
        """Helper function to compare weights between two models."""
        assert len(fsdp1_weights) == len(fsdp2_weights), 'Number of parameters should match between FSDP1 and FSDP2'

        # Note that the names are not guaranteed to be the same, but the order should be the same since
        # we are using the same model.
        for name, fsdp1_weight, fsdp2_weight in zip(
            fsdp2_weights.keys(),
            fsdp1_weights.values(),
            fsdp2_weights.values(),
        ):

            assert fsdp1_weight.shape == fsdp2_weight.shape, \
                f'Shape mismatch for {name}: FSDP1 {fsdp1_weight.shape} vs FSDP2 {fsdp2_weight.shape}'

            assert torch.allclose(fsdp1_weight, fsdp2_weight, atol=tolerance, rtol=tolerance), \
                f'Weight difference for {name} exceeds tolerance.'

    @world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_fsdp2_sync_module_state_error_when_rank_0_has_meta_params(
        self,
        world_size: int,
        device: str,
    ):
        """Test that FSDP2 raises an error when rank 0 has meta parameters and other ranks have CUDA parameters."""
        del world_size
        resolved_device = 'meta' if dist.get_local_rank() == 0 else device
        model = SimpleComposerMLP(num_features=10, device=resolved_device)
        model.add_fsdp_wrap_attribute_to_children()

        with pytest.raises(ValueError) as e:
            create_trainer_with_model(model=model, num_classes=10, use_fsdp2=True)
        assert 'Invalid FSDP2 model initialization detected' in str(e.value)

    @world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_fsdp2_sync_module_state_error_when_multiple_model_references(
        self,
        world_size: int,
        device: str,
    ):
        """Test that FSDP2 raises an error when the model has multiple model references."""
        del world_size
        resolved_device = device if dist.get_local_rank() == 0 else 'meta'
        model = SimpleComposerMLP(num_features=10, device=resolved_device, add_multiple_model_references=True)
        model.add_fsdp_wrap_attribute_to_children()
        with pytest.raises(ValueError) as e:
            create_trainer_with_model(model=model, num_classes=10, use_fsdp2=True)
        assert 'duplicate module references' in str(e.value)

    @world_size(2)
    @pytest.mark.gpu
    # Note that we are testing on a GPU instance just to make sure we can initialize
    # on CPU and then move to GPU.
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_fsdp2_syncs_module_states_when_multiple_devices_are_used(
        self,
        world_size: int,
        device: str,
    ):
        """Test that FSDP2 syncs module states when multiple devices are used."""
        del world_size
        resolved_device = device if dist.get_local_rank() == 0 else 'meta'
        model = self._create_model_with_mixed_init(SimpleComposerMLP, 10, resolved_device)

        # create a dummy param_init_fn that initializes the weights and biases to 1 and 2 respectively
        # and re-initialize the model on rank 0
        def param_init_fn(module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.constant_(module.bias, 2)

        model.param_init_fn = param_init_fn  # type: ignore
        if dist.get_local_rank() == 0:
            model.apply(model.param_init_fn)

        trainer = create_trainer_with_model(
            model=model,
            num_classes=10,
            use_fsdp2=True,
        )
        assert isinstance(trainer.state.fsdp_config, FSDP2Config)
        assert trainer.state.fsdp_config.sync_module_states, 'sync_module_states should be True'  # type: ignore

        module = trainer.state.model.module  # type: ignore
        assert torch.equal(
            module[0].weight.full_tensor(),  # type: ignore
            torch.ones_like(module[0].weight.full_tensor()),  # type: ignore
        )
        assert torch.equal(
            module[-1].weight.full_tensor(),  # type: ignore
            torch.ones_like(module[-1].weight.full_tensor()),  # type: ignore
        )
        assert torch.equal(
            module[0].bias.full_tensor(),  # type: ignore
            torch.full_like(module[0].bias.full_tensor(), 2),  # type: ignore
        )
        assert torch.equal(
            module[-1].bias.full_tensor(),  # type: ignore
            torch.full_like(module[-1].bias.full_tensor(), 2),  # type: ignore
        )

    @world_size(2)
    @pytest.mark.gpu
    # Note that we are testing on a GPU instance just to make sure we can initialize
    # on CPU and then move to GPU.
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    @pytest.mark.parametrize('model_class', [SimpleWeightTiedModel, PartialWeightTiedModel])
    def test_fsdp2_mixed_init_does_not_break_weight_tying(
        self,
        world_size: int,
        device: str,
        model_class: type,
    ):
        """Test that FSDP2 syncs module states and doesn't break weight tying."""
        del world_size
        resolved_device = device if dist.get_local_rank() == 0 else 'meta'
        model = self._create_model_with_mixed_init(model_class, num_features=10, device=resolved_device)

        trainer = create_trainer_with_model(
            model=model,
            num_classes=10,
            use_fsdp2=True,
        )
        assert isinstance(trainer.state.fsdp_config, FSDP2Config)
        assert trainer.state.fsdp_config.sync_module_states, 'sync_module_states should be True'  # type: ignore
        # Check that the weights are correctly tied after training
        trainer.fit()

        fc1 = model.module[0].net[0]  # type: ignore
        fc2 = model.module[0].net[-1]  # type: ignore

        weight_1 = fc1.weight.full_tensor()  # type: ignore
        weight_2 = fc2.weight.full_tensor()  # type: ignore
        assert (fc1.weight is fc2.weight)  # type: ignore
        assert (torch.equal(weight_1, weight_2))

    @world_size(2)
    @pytest.mark.gpu
    # Note that we are testing on a GPU instance just to make sure we can initialize
    # on CPU and then move to GPU.
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    @pytest.mark.parametrize('model_class', [SimpleWeightTiedModel, PartialWeightTiedModel])
    def test_fsdp2_sync_module_state_aligns_with_optimizer_state(
        self,
        world_size: int,
        device: str,
        model_class: type,
    ):
        """Test that FSDP2 syncs module states and doesn't break optimizer state."""
        del world_size
        resolved_device = device if dist.get_local_rank() == 0 else 'meta'
        model = self._create_model_with_mixed_init(model_class, num_features=10, device=resolved_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        trainer = create_trainer_with_model(
            model=model,
            num_classes=10,
            use_fsdp2=True,
            optimizer=optimizer,
        )

        assert isinstance(trainer.state.fsdp_config, FSDP2Config)
        assert trainer.state.fsdp_config.sync_module_states, 'sync_module_states should be True'  # type: ignore
        trainer.fit()

        fc1 = model.module[0].net[0]  # type: ignore
        fc2 = model.module[0].net[-1]  # type: ignore

        assert torch.equal(
            fc1.weight.to_local(),  # type: ignore
            optimizer.param_groups[0]['params'][0].to_local(),  # type: ignore
        )
        assert torch.equal(
            fc2.weight.to_local(),  # type: ignore
            optimizer.param_groups[0]['params'][0].to_local(),  # type: ignore
        )

    @world_size(2)
    @pytest.mark.gpu
    @pytest.mark.parametrize('device', ['cuda', 'cpu'])
    def test_fsdp2_sync_module_states_mixed_init_weight_equivalence(
        self,
        world_size: int,
        device: str,
    ):
        """Test that FSDP2 sync_module_states produces equivalent weights to FSDP1 with mixed initialization."""
        del world_size

        fsdp1_model = self._create_model_with_mixed_init(SimpleComposerMLP, num_features=10, seed=42, device=device)
        fsdp1_weights = self._train_model_and_extract_weights(
            fsdp1_model,
            use_fsdp2=False,
        )

        fsdp2_model = self._create_model_with_mixed_init(SimpleComposerMLP, num_features=10, seed=42, device=device)
        fsdp2_weights = self._train_model_and_extract_weights(
            fsdp2_model,
            use_fsdp2=True,
        )

        self._compare_weights(fsdp1_weights, fsdp2_weights, tolerance=1e-5)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('model_class', [SimpleComposerMLP, SimpleWeightTiedModel, PartialWeightTiedModel])
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_fsdp2_monolithic_checkpoint_save_and_load(
    world_size: int,
    model_class: type,
    tmp_path: pathlib.Path,
    device: str,
):
    """Test FSDP2 monolithic checkpoint saving and loading with proper model initialization."""
    NUM_FEATURES = 10
    NUM_CLASSES = 10
    BATCH_SIZE = 2
    DATASET_SIZE = 16  # 8 batches, 2 batches per rank, 2 ranks = 32 samples

    # Use tmp_path from all ranks to ensure consistency
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = tmp_paths[0]

    save_interval = '1ba'
    save_filename = 'ba{batch}-rank{rank}.pt'
    resume_file = 'ba1-rank{rank}.pt'
    final_checkpoint = 'latest-rank{rank}.pt'
    total_batches_str = '8ba'

    # Create initial model and trainer on CUDA initially for training
    kwargs = {'num_classes': NUM_CLASSES} if model_class == SimpleComposerMLP else {}
    model1 = model_class(num_features=NUM_FEATURES, device='cuda', **kwargs)
    model1.add_fsdp_wrap_attribute_to_children()
    if dist.get_local_rank() == 0:
        model1.apply(model1.param_init_fn)

    # train for 8 batches but save every 1 batch
    trainer1 = create_trainer_with_model(
        model=model1,
        num_classes=NUM_CLASSES,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        max_duration=total_batches_str,
        use_fsdp2=True,
        save_folder=os.path.join(save_folder, 'first'),
        save_filename=save_filename,
        state_dict_type='full',
        save_interval=save_interval,
    )

    trainer1.fit()
    trainer1.close()

    # Create second trainer to load checkpoint with monolithic checkpoint
    # On either CPU/GPU based on the device parameter
    resolved_device = device if dist.get_local_rank() == 0 else 'meta'
    model2 = model_class(num_features=NUM_FEATURES, device=resolved_device, **kwargs)
    model2.add_fsdp_wrap_attribute_to_children()
    resume_path = os.path.join(save_folder, 'first', resume_file)

    # train for 8 batches to measure equality of checkpoints
    trainer2 = create_trainer_with_model(
        model=model2,
        num_classes=NUM_CLASSES,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        max_duration=total_batches_str,
        use_fsdp2=True,
        state_dict_type='full',
        load_monolith_rank0_only=True,
        load_path=resume_path,
        save_folder=os.path.join(save_folder, 'second'),
        save_filename=save_filename,
    )
    trainer2.fit()
    trainer2.close()

    _assert_checkpoints_equivalent(
        os.path.join(save_folder, 'first', final_checkpoint),
        os.path.join(save_folder, 'second', final_checkpoint),
    )


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('mixed_precision', ['DEFAULT', 'PURE', 'FULL'])
@pytest.mark.parametrize('precision', ['amp_fp16', 'amp_bf16'])
def test_mixed_precision_fsdp2_base(
    world_size: int,
    mixed_precision: str,
    precision: str,
):
    """Test FSDP2 with mixed precision settings."""
    del world_size

    model = SimpleComposerMLP(num_features=10, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()

    trainer = create_trainer_with_model(
        model=model,
        use_fsdp2=True,
        precision=precision,
        mixed_precision=mixed_precision,
    )

    trainer.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('mixed_precision', ['PURE', 'DEFAULT', 'FULL'])
@pytest.mark.parametrize('precision', ['amp_fp16', 'amp_bf16'])
def test_mixed_precision_fsdp2_detailed(
    world_size: int,
    mixed_precision: str,
    precision: str,
):
    del world_size

    model = SimpleComposerMLP(num_features=10, device='cuda')
    model.add_fsdp_wrap_attribute_to_children()

    trainer = create_trainer_with_model(
        model=model,
        use_fsdp2=True,
        precision=precision,
        mixed_precision=mixed_precision,
    )

    # Run some more meticulous tests to check the functionality of the mixed precision settings
    str_to_dtype = {
        'amp_fp16': torch.float16,
        'amp_bf16': torch.bfloat16,
    }

    trainer.state.model.train()
    # We don't need to cast the input to the correct dtype because `cast_forward_inputs` in MixedPrecisionPolicy
    # will handle that for us
    test_input = torch.randn(2, 10, device='cuda')

    if mixed_precision == 'PURE':
        expected_param_dtype = str_to_dtype[precision]
        # If param_dtype == reduce_dtype, the reduce_dtype set to None in FSDPParam.py
        expected_reduce_dtype = None
    elif mixed_precision == 'DEFAULT':
        expected_param_dtype = str_to_dtype[precision]
        expected_reduce_dtype = torch.float32
    elif mixed_precision == 'FULL':
        expected_param_dtype = torch.float32
        # MixedPrecisionPolicy.py sets reduce_dtype to None for FULL mixed precision
        expected_reduce_dtype = None

    # Track validation results
    param_dtype_validations = []
    grad_dtype_validations = []

    def validate_all_gathered_params(module, input):
        """Hook to validate all-gathered parameter dtypes during forward pass"""
        try:
            actual_dtype = module.weight.dtype
            new_validation = {
                'module': module.__class__.__name__,
                'expected': expected_param_dtype,
                'actual': actual_dtype,
                'matches': actual_dtype == expected_param_dtype,
            }
            param_dtype_validations.append(new_validation)
        except Exception:
            pass

    def validate_gradient_dtypes(module, input):
        """Hook to validate gradient dtypes during the forward pass. Since the actual all-reduce happens after any
        available hook to register, we just check the validity of the _reduce_dtype of the FSDPParamGroup"""
        try:
            assert isinstance(module, torch.distributed.fsdp.FSDPModule)
            actual_grad_dtype = module._get_fsdp_state()._fsdp_param_group._reduce_dtype  # type: ignore
            new_validation = {
                'module': module.__class__.__name__,
                'expected': expected_reduce_dtype,
                'actual': actual_grad_dtype,
                'matches': actual_grad_dtype == expected_reduce_dtype,
            }
            grad_dtype_validations.append(new_validation)
        except Exception:
            pass

    # Register hooks on all Linear modules (where FSDP2 is applied)
    hook_handles = []
    for _, module in trainer.state.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Forward hook to check all-gathered parameter dtypes (after the all-gather is done; prepend = False)
            handle1 = module.register_forward_pre_hook(validate_all_gathered_params, prepend=False)
            # Another hook to check gradient dtypes (this is a workaround since the regular backward hooks seem to be called too early)
            # TODO: Will need to investigate this further...
            handle2 = module.register_forward_pre_hook(validate_gradient_dtypes, prepend=False)
            hook_handles.extend([handle1, handle2])

    try:
        # Run forward and backward pass to trigger the hooks
        output = trainer.state.model(test_input)
        loss = output.mean()
        loss.backward()

        # Validate parameter dtypes from hooks
        assert len(param_dtype_validations) > 0, 'Should have validated at least one parameter dtype'
        for validation in param_dtype_validations:
            assert validation['matches'], \
                f"Parameter dtype mismatch in {validation['module']}: expected {validation['expected']}, got {validation['actual']}"

        # Validate gradient dtypes from hooks
        assert len(grad_dtype_validations) > 0, 'Should have validated at least one gradient dtype'
        for validation in grad_dtype_validations:
            assert validation['matches'], \
                f"Gradient dtype mismatch in {validation['module']}: expected {validation['expected']}, got {validation['actual']}"

        # Also validate the final output dtype matches param_dtype (computation dtype)
        assert output.dtype == expected_param_dtype, \
            f'Output dtype should be {expected_param_dtype} but got {output.dtype}'

    finally:
        # Clean up hooks
        for handle in hook_handles:
            handle.remove()
