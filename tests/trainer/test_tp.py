# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import (
    RandomClassificationDataset,
    SimpleModel,
    world_size,
)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.skip  # TP does not work with DP dimension 1
@pytest.mark.filterwarnings("ignore:FSDP is switching to use `NO_SHARD`.*:UserWarning")
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
def test_tp_train(world_size: int):
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PrepareModuleInput,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=10)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=dataloader,
        tp_config={
            'layer_plan': layer_plan,
            'tensor_parallel_degree': 2,
        },
        fsdp_config={},
        max_duration='3ba',
    )

    trainer.fit()
