# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader
from torch.distributed._tensor import DTensor

from composer.trainer.trainer import Trainer
from composer.utils import dist, load_checkpoint
from tests.common import (
    PartialWeightTiedModel,
    RandomClassificationDataset,
    SimpleWeightTiedModel,
    world_size,
)
from composer.utils.parallelism import FSDP2Config, ParallelismConfig

SKIP_TEST = version.parse(torch.__version__) < version.parse('2.6.0')
if not SKIP_TEST:
    # TODO move this to top once we decprecate torch 2.5
    from composer.distributed.fsdp2 import prepare_fully_shard
else:
    prepare_fully_shard = lambda *args, **kwargs: None

_INIT_DEVICES = ['cuda', 'meta']


def create_trainer_with_model(
    cls: type,
    device: str,
    num_classes: int = 10,
    max_duration: str = '10ep',
) -> Trainer:
    """Helper function to create a Trainer with a model, dataloader, and FSDP2 configuration."""
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    
    model = cls(num_features=num_classes, device=device)

    fsdp2_config = FSDP2Config(
        device_mesh=None,
        reshard_after_forward=True,
    )
    prepare_fully_shard(model=model.module, fsdp2_config=fsdp2_config)
    # NOTE module to_empty should only happen after the model is fully sharded and parameters are coverted to Dtensor
    # otherwise to_empty breaks weight tying
    # TODO we should guardrail this in prepare_fully_shard
    model.to_empty(device='cuda')
    for module in model.modules():
        model.param_init_fn(module)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        max_duration=max_duration,
        parallelism_config=ParallelismConfig(fsdp2=fsdp2_config),
    )
    return trainer


@pytest.mark.parametrize('model_class', [SimpleWeightTiedModel, PartialWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')
@pytest.mark.skipif(SKIP_TEST, reason='FSDP2 is not available in torch < 2.6.0')
def test_fsdp2_initialization_with_tied_params(
    model_class: type,
    device: str,
    world_size: int,
):
    """test FSDP2 initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied.
    """
    trainer = create_trainer_with_model(
        cls=model_class,
        device=device,
    )

    # Initialization checks
    model = trainer.state.model
    assert isinstance(model, SimpleWeightTiedModel | PartialWeightTiedModel), f'Expected model to be SimpleWeightTiedModel or PartialWeightTiedModel, got {type(model)}'
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
@pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')
@pytest.mark.skipif(SKIP_TEST, reason='FSDP2 is not available in torch < 2.6.0')
def test_fsdp2_checkpointing(
    model_class: type,
    device: str,
    world_size: int,
    tmp_path: pathlib.Path,
):
    """Test FSDP2 checkpointing and weight tying after loading."""
    trainer = create_trainer_with_model(
        cls=model_class,
        device=device,
    )

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
    trainer = create_trainer_with_model(
        cls=model_class,
        device=device,
    )
    load_checkpoint(str(pathlib.Path(ckpt_path).parent), trainer.state, trainer.logger, load_weights_only=True)

    model = trainer.state.model
    assert isinstance(model, SimpleWeightTiedModel), f'Expected model to be SimpleWeightTiedModel, got {type(model)}'
    assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
    assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'
    # Check that the weights are still tied after loading and that the local weights are the same
    assert torch.equal(weight_1_local, model.mlp.fc1.weight.to_local())
    assert torch.equal(weight_2_local, model.mlp.fc2.weight.to_local())
    assert model.mlp.fc1.weight is model.mlp.fc2.weight


@pytest.mark.skipif(SKIP_TEST, reason='FSDP2 is not available in torch < 2.6.0')
@pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')
def test_fsdp2_config():
    """Test that FSDP2Config read-only properties work as expected."""
    if not SKIP_TEST:
        # Create a config instance
        config = FSDP2Config()
        
        # Test reading properties (should succeed)
        assert config.auto_wrap is True
        assert config.load_monolith_rank0_only is False
        assert config.sync_module_states is False
        assert config.activation_cpu_offload is False
        assert config.data_parallel_shard_degree == -1
        assert config.data_parallel_replicate_degree is None
        assert config.state_dict_type == 'sharded'
        assert config.use_orig_params is True
        
        # Test setting properties (should fail)
        read_only_props = [
            ("auto_wrap", False),
            ("load_monolith_rank0_only", True),
            ("sync_module_states", True),
            ("activation_cpu_offload", True),
            ("data_parallel_shard_degree", 2),
            ("data_parallel_replicate_degree", 2),
            ("state_dict_type", "full"),
            ("use_orig_params", False)
        ]
        
        for prop, value in read_only_props:
            with pytest.raises(AttributeError):
                setattr(config, prop, value)
        
        # Test that core properties can be set
        config.device_mesh = None
        config.reshard_after_forward = False
        assert config.device_mesh is None
        assert config.reshard_after_forward is False

