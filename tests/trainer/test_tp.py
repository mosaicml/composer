# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.utils import reproducibility
from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import (
    RandomClassificationDataset,
    SimpleModel,
    SimpleComposerMLP,
    SimpleDataset,
    world_size,
)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_train(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))

    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader,
        parallelism_config={
            'tp': {
                'layer_plan': layer_plan,
                'tensor_parallel_degree': 2,
            },
            'fsdp': {},
        },
        max_duration='3ba',
    )

    trainer.fit()


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_with_param_groups(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD([{
        'params': model.fc1.parameters(),
        'lr': 0.1,
    }, {
        'params': model.fc2.parameters(),
        'lr': 0.5,
    }])

    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }

    expected_error = 'Multiple optimizer groups are not supported with tensor parallelism.'

    with pytest.raises(RuntimeError, match=expected_error):
        _ = Trainer(
            model=model,
            optimizers=optimizer,
            train_dataloader=dataloader,
            parallelism_config={
                'tp': {
                    'layer_plan': layer_plan,
                    'tensor_parallel_degree': 2,
                },
                'fsdp': {},
            },
            max_duration='3ba',
        )


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_with_subset_of_params(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel

    model = SimpleModel()
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))
    optimizer = torch.optim.SGD(model.fc1.parameters(), lr=0.1)

    layer_plan = {
        'fc1': ColwiseParallel(),
    }

    expected_error = 'Passing in a subset of model parameters to the optimizer is not supported with tensor parallelism.'

    with pytest.raises(ValueError, match=expected_error):
        _ = Trainer(
            model=model,
            optimizers=optimizer,
            train_dataloader=dataloader,
            parallelism_config={
                'tp': {
                    'layer_plan': layer_plan,
                    'tensor_parallel_degree': 2,
                },
                'fsdp': {},
            },
            max_duration='3ba',
        )




@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_correctness(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    import icecream
    icecream.install()

    def train_and_fit(parallelism_config):
        """Train a simple model with different parallelism_configs."""
        num_features, num_classes, batch_size, size, seed = 64, 10, 8, 32, 42
        reproducibility.seed_all(seed)

        dataset = RandomClassificationDataset(shape=(num_features,), num_classes=num_classes, size=size) # X=(num_features,), y=(,), i.e. scalar
        dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=batch_size) # X=(batch_size, num_features), y=(batch_size,)
        model = SimpleComposerMLP(num_features=num_features, device='cuda', num_classes=num_classes)

        trainer = Trainer(
            seed=seed,
            device='gpu',
            model=model,
            max_duration='1ba',
            train_dataloader=dataloader,
            parallelism_config=parallelism_config,
            callbacks=[MemoryMonitor()],
            loggers=[InMemoryLogger()],
            )
        trainer.fit()
        ic(dataset.y)

        log = trainer.logger.destinations[0].most_recent_values
        stats = {
            'loss':log['loss/train/total'],
            'accuracy': log['metrics/train/MulticlassAccuracy'].item(),
            'peak_memory': log['memory/peak_reserved_mem'],
            'parameters': trainer.state.model.parameters(),
        }
        return stats

    # DDP
    stats_ddp = train_and_fit(parallelism_config=None)
    ic(stats_ddp)

    # FSDP + TP
    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': 2}
    parallelism_config = {'fsdp': {}, 'tp': tp_config}
    stats_fsdp_tp = train_and_fit(parallelism_config=parallelism_config)
    ic(stats_fsdp_tp)


#    # forward pass with FSDP and no TP
#    model_fsdp, dataloader_fsdp = _helper()
#    trainer_fsdp = Trainer(
#        seed=SEED,
#        model=model_fsdp,
#        parallelism_config={'fsdp': {}},
#        # callbacks=[MemoryMonitor()],
#        # loggers=[InMemoryLogger()],
#        )
#    outputs_fsdp = torch.stack(trainer_fsdp.predict(dataloader_fsdp))


#    # forward pass with FSDP and TP
#    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
#    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': 2}
#    model_fsdp_tp, dataloader_fsdp_tp = _helper()
#    trainer_fsdp_tp = Trainer(
#        seed=SEED,
#        model=model_fsdp_tp,
#        max_duration='1ba',
#        train_dataloader=dataloader_fsdp_tp,
#        # callbacks=[MemoryMonitor()],
#        loggers=[InMemoryLogger()],
#        parallelism_config={'fsdp': {}, 'tp': tp_config},
#        )
#    trainer_fsdp_tp.fit()


#    # match shape
#    assert outputs.shape == outputs_fsdp.shape
#    assert outputs.shape == outputs_fsdp_tp.shape


#    # match elements
#    assert torch.allclose(outputs, outputs_fsdp)
#    assert torch.allclose(outputs, outputs_fsdp_tp)
