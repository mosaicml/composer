# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.core.state import fsdp_get_optim_state_dict, fsdp_state_dict_type_context
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
from tests.trainer.test_fsdp_checkpoint import _compare_model_params_between_state_dicts


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


def get_mono_state_dict_from_sharded_one(trainer):
        state_dict = trainer.state.state_dict()
        state_dict.pop('optimizers')
        state_dict.pop('model')

        # Add in unsharded model params.
        with fsdp_state_dict_type_context(trainer.state.model, state_dict_type='full'):
            state_dict['model'] = trainer.state.model.state_dict()

        optimizer = trainer.state.optimizers[0]
        state_dict['optimizers'] = {
            type(optimizer).__qualname__:
                fsdp_get_optim_state_dict(trainer.state.model, optimizer, state_dict_type='full'),
        }
        return state_dict


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_correctness(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    def get_trainer(parallelism_config):
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
        return trainer

    # DDP
    trainer_ddp = get_trainer(parallelism_config=None)
    out_ddp = torch.stack(trainer_ddp.predict(trainer_ddp.state.train_dataloader, subset_num_batches=1))

    # FSDP
    fsdp_config = {'state_dict_type': 'sharded'} # {'data_parallel_shard_degree': 2}
    trainer_fsdp = get_trainer(parallelism_config={'fsdp': fsdp_config})
    out_fsdp = torch.stack(trainer_fsdp.predict(trainer_fsdp.state.train_dataloader, subset_num_batches=1))

    # FSDP + TP
    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = {'layer_plan': layer_plan, 'tensor_parallel_degree': 2}
    fsdp_config = {'state_dict_type': 'sharded'}
    parallelism_config = {'fsdp': fsdp_config, 'tp': tp_config}
    trainer_fsdp_tp = get_trainer(parallelism_config=parallelism_config)
    out_fsdp_tp = torch.stack(trainer_fsdp_tp.predict(trainer_fsdp_tp.state.train_dataloader, subset_num_batches=1))

    assert out_ddp.shape == out_fsdp.shape == out_fsdp_tp.shape
    assert torch.allclose(out_ddp, out_fsdp)
    assert torch.allclose(out_ddp, out_fsdp_tp), f"Outputs have different values: {out_ddp=} and {out_fsdp_tp=}"
