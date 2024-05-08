# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import (
    RandomClassificationDataset,
    SimpleModel,
    world_size,
)

@pytest.mark.gpu
@world_size(2)
def test_tp_train(world_size: int):
    model = SimpleModel()
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    device_mesh = init_device_mesh(
        'cuda',
        (2,),
        mesh_dim_names=('tensor_parallel',),
    )

    layer_plan = {
        'model.fc1': ColwiseParallel(),
        'model.fc2': RowwiseParallel(),
    }

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        tp_config={
            'device_mesh': device_mesh,
            'layer_plan': layer_plan,
        },
        max_duration='3ba',
    )

    trainer.fit()
