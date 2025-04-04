# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader

from composer.distributed.dist_strategy import prepare_fully_shard
from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist
from composer.utils.parallelism import FSDP2Config
from tests.common import (
    PartialWeightTiedModel,
    RandomClassificationDataset,
    SimpleWeightTiedModel,
    world_size,
)

_INIT_DEVICES = ['cuda']


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, PartialWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
def test_fsdp2_initialization_with_tied_params(
    model: ComposerClassifier,
    world_size: int,
    device: str,
):
    """test FSDP2 initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied.
    """
    num_classes = 10

    resolved_device = device
    model = model(num_features=num_classes, device=resolved_device)
    assert isinstance(model, (SimpleWeightTiedModel, PartialWeightTiedModel))
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    fsdp2_config = FSDP2Config(
        device_mesh=None,
        reshard_after_forward=True,
        mp_policy=None,
        offload_policy=None,
    )
    prepare_fully_shard(model=model.module, fsdp2_config=fsdp2_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        max_duration='10ep',
    )
    trainer.fit()
    assert len(model.mlp._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the mlp module'
    assert len(model.mlp.fc1._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc1 module'
    assert len(model.mlp.fc2._forward_pre_hooks) == 0, 'Expected 0 forward pre-hook on the fc2 module'
    assert len(model.module._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the root module'
    weight_1 = model.mlp.fc1.weight.full_tensor()  # type: ignore[reportAttributeAccessIssue]
    weight_2 = model.mlp.fc2.weight.full_tensor()  # type: ignore[reportAttributeAccessIssue]
    assert (model.mlp.fc1.weight is model.mlp.fc2.weight)
    assert (torch.equal(weight_1, weight_2))

    if isinstance(model, PartialWeightTiedModel):
        assert len(model.fc3._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the fc3 module'
