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
    EmbeddedWeightTiedModel,
    RandomClassificationDataset,
    SimpleWeightTiedModel,
    world_size,
)

_INIT_DEVICES = ['cuda']


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, EmbeddedWeightTiedModel])
@pytest.mark.parametrize('device', _INIT_DEVICES)
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:The passed in model appears to have tied weights.*:UserWarning')
def test_fsdp_device_initialization(
    model: ComposerClassifier,
    world_size: int,
    device: str,
):
    """test FSDP device initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied. This test also covers both 'cpu' and
    'meta' devices. This is because 'meta' will result in deferred initialization until FSDP is initialized

    """
    num_classes = 10

    resolved_device = device
    model = model(num_features=num_classes, device=resolved_device)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))

    fsdp2_config = FSDP2Config(
        device_mesh=None,
        reshard_after_forward=True,
        mp_policy=None,
        offload_policy=None,
    )
    prepare_fully_shard(model=model.module, fsdp2_config=fsdp2_config)
    # if dist.get_global_rank() == 0:
    #     print(model)
    #     for name, child in model.named_children():
    #         print(f'Child: {name}')
    #         print(child._forward_pre_hooks)
    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        max_duration='3ba',
    )

    trainer.fit()
    if isinstance(model, SimpleWeightTiedModel):
        assert len(model.mlp._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the mlp module'
        assert len(model.mlp.fc1._forward_hooks) == 0, 'Expected 0 forward hook on the fc1 module'
        assert len(model.mlp.fc2._forward_hooks) == 0, 'Expected 0 forward hook on the fc2 module'
        assert len(model.module._forward_hooks) == 1, 'Expected 1 forward hook on the root module'
        weight_1 = model.mlp.fc1.weight.full_tensor()
        weight_2 = model.mlp.fc2.weight.full_tensor()
        assert (model.mlp.fc1.weight is model.mlp.fc2.weight)
        assert (torch.equal(weight_1, weight_2))

    if isinstance(model, EmbeddedWeightTiedModel):
        assert len(model.module._forward_pre_hooks) == 1, 'Expected 1 forward pre-hook on the root module'
        assert len(model.net1._forward_pre_hooks) == 0, 'Expected 0 forward hook on the net1 module'
        assert len(model.net2._forward_hooks) == 0, 'Expected 0 forward hook on the net2 module'
        weight_1 = model.net1.fc1.weight.full_tensor()
        weight_2 = model.net2.fc1.weight.full_tensor()
        assert (model.net1.fc1.weight is model.net2.fc1.weight)
        assert (torch.equal(weight_1, weight_2))
