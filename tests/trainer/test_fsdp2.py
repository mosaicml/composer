# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist
from composer.utils.parallelism import FSDP2Config
from composer.distributed.dist_strategy import prepare_fully_shard
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
        ignored_params=None,
    )
    prepare_fully_shard(model=model, fsdp2_config=fsdp2_config)

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
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