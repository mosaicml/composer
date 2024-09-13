# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch
import numpy as np
from packaging import version
from torch.utils.data import DataLoader
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import Replicate, Shard

from composer.utils import reproducibility, FSDPConfig, TPConfig, ParallelismConfig, dist
from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from tests.common import (
    RandomClassificationDataset,
    SimpleModel,
    SimpleComposerMLP,
    world_size,
)

from icecream import ic

@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_train(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    # Normally, each TP rank receives the same data via data replication
    # In this test, we do not do this: each TP rank gets different data
    # This is okay - we are testing the TP mechanism, not actual TP correctness
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
            'tp': TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2),
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

    # Normally, each TP rank receives the same data via data replication
    # In this test, we do not do this: each TP rank gets different data
    # This is okay - we are testing the TP mechanism, not actual TP correctness
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
                'tp': TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2),
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

    # Normally, each TP rank receives the same data via data replication
    # In this test, we do not do this: each TP rank gets different data
    # This is okay - we are testing the TP mechanism, not actual TP correctness
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
                'tp': TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2),
                'fsdp': {},
            },
            max_duration='3ba',
        )

class GatherColwiseParallel(ColwiseParallel):
    """ColwiseParallel layer that all-gathers the inputs first."""
    def __init__(
        self,
        *,
        use_local_output: bool = True
    ):
        super().__init__()
        # Inputs over the TP dimension are sharded by device batches.
        self.input_layouts = (Shard(0), )
        # All-gather inputs so that each GPU now has the same input activations.
        self.desired_input_layouts = (Replicate(), )
        # self.output_layouts = (Shard(-1), )
        self.use_local_output = use_local_output


def get_trainer(
    parallelism_config: Optional[ParallelismConfig] = None,
    size: int = 4,
    batch_size: int = 4,
    num_classes: int = 2,
    num_features: int = 6,
    seed: int = 42,
    device: str = 'cuda',
    ):
    """Trainer for a simple model with any parallelism_config."""

    reproducibility.seed_all(seed)
    dataset = RandomClassificationDataset(shape=(num_features,), num_classes=num_classes, size=size, device=device) # X=(num_features,), y=(,), i.e. scalar
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=batch_size) # X=(batch_size, num_features), y=(batch_size,)
    model = SimpleComposerMLP(num_features=num_features, device=device, num_classes=num_classes)

    trainer = Trainer(
        seed=seed,
        device='gpu',
        model=model,
        max_duration='1ep',
        train_dataloader=dataloader,
        parallelism_config=parallelism_config,
        callbacks=[MemoryMonitor()],
        loggers=[InMemoryLogger()],
        )
    return trainer


def _forward(trainer):
    batch = next(iter(trainer.state.train_dataloader))
    output = trainer.state.model.forward(batch)
    return output


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_forward(world_size: int):
    """Test that the forward pass with DDP, FSDP, FSDP + TP all output the same tensor."""

    # DDP forward pass
    ddp_trainer = get_trainer()
    ddp_out = _forward(ddp_trainer)

    # FSDP forward pass
    fsdp_config = FSDPConfig(
        state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        mixed_precision='full',
        )
    parallelism_config = ParallelismConfig(fsdp=fsdp_config)
    fsdp_trainer = get_trainer(parallelism_config=parallelism_config)
    fsdp_out = _forward(fsdp_trainer)

    # FSDP + TP forward pass
    layer_plan = {
        'fc1': GatherColwiseParallel(),
        'fc2': RowwiseParallel(output_layouts=Shard(0)),
        }
    tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
    parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)
    tp_fsdp_trainer = get_trainer(parallelism_config=parallelism_config)
    tp_fsdp_out = _forward(tp_fsdp_trainer)

    assert ddp_out.shape == fsdp_out.shape == tp_fsdp_out.shape, f"Outputs have different shapes: {ddp_out.shape=}, {fsdp_out.shape=}, {tp_fsdp_out.shape=}"
    assert torch.allclose(ddp_out, fsdp_out, atol=1e-3), f"Outputs have different values: {ddp_out=} and {fsdp_out=}"
    assert torch.allclose(ddp_out, tp_fsdp_out, atol=1e-3), f"Outputs have different values: {ddp_out=} and {tp_fsdp_out=}"


def _get_stats(trainer: Trainer) -> dict[str, np.ndarray]:
    logger = trainer.logger.destinations[0]
    stats = {
        'loss_array': logger.get_timeseries('loss/train/total')['loss/train/total'],
        'accuracy_array': logger.get_timeseries('metrics/train/MulticlassAccuracy')['metrics/train/MulticlassAccuracy'],
        # 'peak_reserved_mem': logger.get_timeseries('memory/peak_reserved_mem')['memory/peak_reserved_mem'],
    }
    return stats


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fit(world_size: int):
    """Test that trainer.fit() with DDP, FSDP, FSDP + TP all output the same loss and accuracy."""
    import warnings
    from icecream import install
    install()
    warnings.filterwarnings("ignore")

    size = 1024

    # DDP forward pass
    ddp_trainer = get_trainer(parallelism_config=None, size=size)
    ddp_trainer.fit()
    # ddp_trainer.close()
    ddp_stats = _get_stats(ddp_trainer)

    # FSDP forward pass
    fsdp_config = FSDPConfig(
        state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        mixed_precision='full',
        )
    fsdp_trainer = get_trainer(parallelism_config={'fsdp': fsdp_config}, size=size)
    fsdp_trainer.fit()
    fsdp_stats = _get_stats(fsdp_trainer)

    # FSDP + TP forward pass
    layer_plan = {
        'fc1': GatherColwiseParallel(),
        'fc2': RowwiseParallel(output_layouts=Shard(0)),
        }
    tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
    parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)
    tp_fsdp_trainer = get_trainer(parallelism_config=parallelism_config, size=size)
    tp_fsdp_trainer.fit()
    tp_fsdp_stats = _get_stats(tp_fsdp_trainer)

    ic(ddp_stats)
    ic(fsdp_stats)
    ic(tp_fsdp_stats)

    # # assert ddp_out.shape == fsdp_out.shape == tp_fsdp_out.shape, f"Outputs have different shapes: {ddp_out.shape=}, {fsdp_out.shape=}, {tp_fsdp_out.shape=}"
    # # assert torch.allclose(ddp_out, fsdp_out, atol=1e-3), f"Outputs have different values: {ddp_out=} and {fsdp_out=}"
    # assert torch.allclose(ddp_out, tp_fsdp_out, atol=1e-3), f"Outputs have different values: {ddp_out=} and {tp_fsdp_out=}"
