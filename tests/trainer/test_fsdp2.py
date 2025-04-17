# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import (
    PartialWeightTiedModel,
    RandomClassificationDataset,
    SimpleWeightTiedModel,
    world_size,
)

SKIP_TEST = version.parse(torch.__version__) < version.parse('2.6.0')
if not SKIP_TEST:
    # TODO move this to top once we decprecate torch 2.5
    from torch.distributed.tensor import DTensor

    from composer.distributed.fsdp2 import FSDP2Config, prepare_fully_shard

_INIT_DEVICES = ['cuda', 'meta']


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, PartialWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')
@pytest.mark.skipif(SKIP_TEST, reason='FSDP2 is not available in torch < 2.6.0')
def test_fsdp2_initialization_with_tied_params(
    model: ComposerClassifier,
    world_size: int,
    device: str,
):
    """test FSDP2 initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied.
    """
    num_classes = 10

    model = model(num_features=num_classes, device=device)
    assert isinstance(model, (SimpleWeightTiedModel, PartialWeightTiedModel))
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    fsdp2_config = FSDP2Config(
        device_mesh=None,
        reshard_after_forward=True,
        mp_policy=None,
        offload_policy=None,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    prepare_fully_shard(model=model.module, optimizers=optimizer, fsdp2_config=fsdp2_config)

    # Initialization checks
    assert len(model.mlp._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the mlp module'
    assert len(model.mlp.fc1._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc1 module'
    assert len(model.mlp.fc2._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc2 module'
    assert len(model.module._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the root module'
    assert isinstance(model.mlp.fc1.weight, DTensor), 'mlp.fc1.weight should be a DTensor'
    assert isinstance(model.mlp.fc2.weight, DTensor), 'mlp.fc2.weight should be a DTensor'

    if isinstance(model, PartialWeightTiedModel):
        assert len(model.fc3._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the fc3 module'
    assert model.mlp.fc1.weight.size(0) == model.mlp.fc2.weight.to_local(
    ).size(0) * world_size, 'Expect global weight size to be equal to local weight size * world_size on dim 0'

    # manual param init
    # TODO remove this once we integrate param init into fsdp2 helper functions
    model.to_empty(device='cuda')
    for module in model.modules():
        model.param_init_fn(module)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        max_duration='10ep',
    )
    trainer.fit()

    # Check that the weights are correctly tied
    weight_1 = model.mlp.fc1.weight.full_tensor()
    weight_2 = model.mlp.fc2.weight.full_tensor()
    assert (model.mlp.fc1.weight is model.mlp.fc2.weight)
    assert (torch.equal(weight_1, weight_2))


@pytest.mark.parametrize('case', ['all_params_one_group', 'subset_one_group', 'multiple_groups'])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:FSDP2 Config/APIs are experimental*:UserWarning')
@pytest.mark.skipif(SKIP_TEST, reason='FSDP2 is not available in torch < 2.6.0')
def test_fsdp2_optimizer_handling(
    case: str,
    world_size: int,
    device: str,
):
    """Test FSDP2 correctly updates optimizer state for various configurations."""
    del world_size

    num_classes = 10
    model = PartialWeightTiedModel(num_features=num_classes, device=device)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=10, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))

    all_params_list = list(model.parameters())
    fc1_params_list = list(model.mlp.fc1.parameters())
    fc3_params_list = list(model.fc3.parameters())
    remaining_params_list = [
        p for p in all_params_list
        if not any(p is param for param in fc1_params_list)
        and not any(p is param for param in fc3_params_list)
        and not any(p is param for param in model.mlp.fc2.parameters()) # To avoid double counting the tied parameter
    ]

    if case == 'all_params_one_group':
        optimizer_input = [{'params': all_params_list, 'lr': 0.01}]
    elif case == 'subset_one_group':
        optimizer_input = [{'params': fc1_params_list, 'lr': 0.02}]
    elif case == 'multiple_groups':
        optimizer_input = [
            {'params': fc1_params_list, 'lr': 0.01},
            {'params': fc3_params_list, 'lr': 0.02},
            {'params': remaining_params_list, 'lr': 0.03},
        ]

    optimizer = torch.optim.Adam(optimizer_input)

    fsdp2_config = FSDP2Config(
        device_mesh=None,
        reshard_after_forward=True,
        mp_policy=None,
        offload_policy=None,
    )
    prepare_fully_shard(model=model.module, optimizers=optimizer, fsdp2_config=fsdp2_config)

    def validate_optimizer_state(current_optimizer: torch.optim.Optimizer, stage: str):
        assert len(current_optimizer.param_groups) == len(optimizer_input), \
            f"[{case}/{stage}] Group count mismatch. Expected {len(optimizer_input)}, Got {len(current_optimizer.param_groups)}"
        for i, group in enumerate(current_optimizer.param_groups):
            opt_params = group['params']
            assert len(opt_params) == len(optimizer_input[i]['params']), \
                f"[{case}/{stage}] Group {i}: Param count mismatch. Expected {len(optimizer_input[i]['params'])}, Got {len(opt_params)}"
            assert all(isinstance(p, DTensor) for p in opt_params), \
                f"[{case}/{stage}] Group {i}: Not all parameters are DTensors"
            assert group['lr'] == optimizer_input[i]['lr'], \
                f"[{case}/{stage}] Group {i}: LR mismatch. Expected {optimizer_input[i]['lr']}, Got {group['lr']}"

    validate_optimizer_state(optimizer, stage="after_fully_shard")

    model.to_empty(device='cuda')
    for module in model.modules():
        model.param_init_fn(module)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        max_duration='1ba',
    )
    trainer.fit()

    validate_optimizer_state(optimizer, stage="after_fit")
