# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.models import ComposerClassifier
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import EmbeddedWeightTiedModel, RandomClassificationDataset, SimpleWeightTiedModel


@pytest.mark.parametrize('model', [SimpleWeightTiedModel, EmbeddedWeightTiedModel])
@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.gpu
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_device_initialization(model: ComposerClassifier, device: str):
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
        fsdp_config={},
        max_duration='3ba',
    )

    trainer.fit()
    if isinstance(model, SimpleWeightTiedModel):
        weight_1 = model.mlp.fc1.weight
        weight_2 = model.mlp.fc2.weight
        assert (id(weight_1) == id(weight_2))
        assert (torch.equal(weight_1, weight_2))

    if isinstance(model, EmbeddedWeightTiedModel):
        weight_1 = model.net1.fc1.weight
        weight_2 = model.net2.fc1.weight
        assert (id(weight_1) == id(weight_2))
        assert (torch.equal(weight_1, weight_2))
