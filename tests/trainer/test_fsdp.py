# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import (EmbeddedWeightTiedModel, RandomClassificationDataset, SimpleModel, SimpleWeightTiedModel,
                          world_size)


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, EmbeddedWeightTiedModel])
@pytest.mark.parametrize('mixed_precision', ['FULL', 'DEFAULT', 'PURE'])
@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.parametrize('reentrant', [True, False])
@world_size(2)
@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:The passed in model appears to have tied weights.*:UserWarning')
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='FSDP requires PyTorch 1.13 or higher')
def test_fsdp_device_initialization(model: ComposerClassifier, mixed_precision: str, reentrant: bool, world_size: int,
                                    device: str):
    """test FSDP device initialization for a simple model with weight tying and a model where two modules
    from separate submodules have weight tying applied. This test also covers both 'cpu' and
    'meta' devices. This is because 'meta' will result in deferred initialization until FSDP is initialized

    """
    num_classes = 10
    model = model(num_features=num_classes, device=device)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'activation_checkpointing_reentrant': reentrant,
            'mixed_precision': mixed_precision
        },
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


@pytest.mark.parametrize('model', [SimpleModel])
@pytest.mark.parametrize('mixed_precision', ['FULL', 'DEFAULT', 'PURE'])
@pytest.mark.gpu
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='FSDP requires PyTorch 1.13 or higher')
def test_fsdp_meta_initialization_none(model: ComposerClassifier, mixed_precision: 'str', world_size: int):
    """
    This test is intended to test FSDP for meta initialization when there are attributes
    that are `None` and ensure we don't raise nasty UserWarnings.
    """
    num_classes = 2
    model = model(num_features=1, num_classes=num_classes, device='meta', bias=False)
    dataset = RandomClassificationDataset(shape=(num_classes,), size=2, num_classes=num_classes)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'mixed_precision': mixed_precision,
            'sharding_strategy': 'NO_SHARD'
        },
        max_duration='3ba',
    )


@pytest.mark.parametrize('forward_prefetch_limit', [1, 2])
@pytest.mark.parametrize('backward_prefetch_limit', [1, 2])
@pytest.mark.gpu
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='FSDP requires PyTorch 1.13 or higher')
def test_fsdp_prefetch_limit(forward_prefetch_limit: int, backward_prefetch_limit: int, world_size: int):
    model = SimpleModel()
    model.fc1._fsdp_wrap = True
    model.fc2._fsdp_wrap = True
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        fsdp_config={
            'forward_prefetch_limit': forward_prefetch_limit,
            'backward_prefetch_limit': backward_prefetch_limit,
        },
        max_duration='3ba',
    )

    trainer.fit()
