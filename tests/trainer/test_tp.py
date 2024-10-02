# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Any, Optional, Union

import numpy as np
import pytest
import torch
from packaging import version
from torch.distributed._tensor import Replicate
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledSGDW
from composer.trainer.trainer import Trainer
from composer.utils import FSDPConfig, ParallelismConfig, TPConfig, dist, reproducibility
from tests.common import (
    RandomClassificationDataset,
    RandomClassificationDatasetReplicated,
    SimpleModel,
    TPSimpleComposerMLP,
    deep_compare,
    world_size,
)


def get_base_trainer(
    parallelism_config: Optional[ParallelismConfig] = None,
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 44,
    device: Union[torch.device, str] = 'cuda',
    replication: Optional[int] = None,
):
    """Trainer for a simple model with any parallelism_config."""

    reproducibility.seed_all(seed)
    if isinstance(device, str):
        device = torch.device(device)

    dataset: Dataset = RandomClassificationDatasetReplicated(
        shape=(num_features,),
        num_classes=num_classes,
        size=size,
        device=device,
        replication=replication,
    )  # X=(num_features,), y=(,), i.e. scalar

    dataloader = DataLoader(
        dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=batch_size,
    )  # X=(batch_size, num_features), y=(batch_size,)

    model = TPSimpleComposerMLP(num_features=num_features, device=device, num_classes=num_classes)

    trainer = Trainer(
        seed=seed,
        device='gpu',
        model=model,
        max_duration='1ep',
        train_dataloader=dataloader,
        precision='fp32',
        parallelism_config=parallelism_config,
        callbacks=[MemoryMonitor()],
        loggers=[InMemoryLogger()],
        progress_bar=False,
        log_to_console=False,
    )

    return trainer


def get_trainer(
    parallelism_strategy: str,
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 44,
    device: Union[torch.device, str] = 'cuda',
    replication: Optional[int] = None,
) -> Trainer:

    if parallelism_strategy == 'ddp':
        parallelism_config = None
    elif parallelism_strategy == 'fsdp':
        fsdp_config = FSDPConfig(
            state_dict_type='full',
            sharding_strategy='SHARD_GRAD_OP',
            mixed_precision='full',
            use_orig_params=True,
        )
        parallelism_config = ParallelismConfig(fsdp=fsdp_config)
    elif parallelism_strategy == 'tp-fsdp':
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        fsdp_config = FSDPConfig(
            state_dict_type='full',
            sharding_strategy='SHARD_GRAD_OP',
            mixed_precision='full',
            use_orig_params=True,
        )
        layer_plan = {
            'fc1': ColwiseParallel(),
            'fc2': RowwiseParallel(),
        }
        tp_config = TPConfig(
            layer_plan=layer_plan,
            tensor_parallel_degree=1 if replication is None else replication,
        )
        parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)
    else:
        raise ValueError(
            f'`parallelism_strategy` must be one of `ddp`, `fsdp`, `tp-fsdp` but was {parallelism_strategy=}.',
        )

    trainer = get_base_trainer(
        size=size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_features=num_features,
        seed=seed,
        device=device,
        replication=replication,
        parallelism_config=parallelism_config,
    )

    return trainer


def forward_pass(trainer):
    reproducibility.seed_all(trainer.state.seed)
    batch = next(iter(trainer.state.train_dataloader))
    output = trainer.state.model.forward(batch)
    return output


def _replace_state_dict_name(state_dict: dict[str, Any], old_name: str, new_name: str) -> dict[str, Any]:
    keys = list(state_dict.keys())
    for key in keys:
        if old_name in key:
            new_key = key.replace(old_name, new_name, 1)
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def compare_models(
    ddp_trainer: Trainer,
    fsdp_trainer: Trainer,
    tp_fsdp_trainer: Trainer,
    check_grad: bool = False,
    atol: float = 0.0,
    rtol: float = 0.0,
):

    # Normally, we compare various models by their state_dict().
    # However, calling `tp_fsdp_trainer.state.state_dict()` directly causes a NCCL timeout
    # due to this pytorch bug: https://github.com/pytorch/pytorch/issues/134095/.
    # As a workaround, we use `tp_fsdp_trainer.state.model.named_parameters()` instead.
    # This issues only exists with `tp_fsdp_trainer.state.state_dict()` and does not
    # arise when calling `ddp_trainer.state.state_dict()` or `fsdp_trainer.state.state_dict()`.
    with FSDP.summon_full_params(fsdp_trainer.state.model, with_grads=check_grad):
        with FSDP.summon_full_params(tp_fsdp_trainer.state.model, with_grads=check_grad):
            ddp_params = dict(ddp_trainer.state.model.named_parameters())
            fsdp_params = dict(fsdp_trainer.state.model.named_parameters())
            tp_fsdp_params = dict(tp_fsdp_trainer.state.model.named_parameters())

            # patch the state dict names:
            #   - ddp adds an extra 'module.' to all param names
            #   - fsdp adds an extra '_fsdp_wrapped_module.' to all param names
            #   - tp-fsdp adds an extra '_fsdp_wrapped_module.' to all param names
            ddp_params = _replace_state_dict_name(ddp_params, 'module.', '')
            fsdp_params = _replace_state_dict_name(fsdp_params, '_fsdp_wrapped_module.', '')
            tp_fsdp_params = _replace_state_dict_name(tp_fsdp_params, '_fsdp_wrapped_module.', '')

            # check grad
            if check_grad:

                def get_grads(params):
                    return {name: param.grad for name, param in params.items()}

                ddp_params = get_grads(ddp_params)
                fsdp_params = get_grads(fsdp_params)
                tp_fsdp_params = get_grads(tp_fsdp_params)

            # collect tensors from different ranks for comparison
            tp_fsdp_params = {
                name: param.redistribute(device_mesh=param.device_mesh, placements=[Replicate()]).to_local()
                for name, param in tp_fsdp_params.items()
            }

            deep_compare(ddp_params, fsdp_params, atol=atol, rtol=rtol)
            deep_compare(tp_fsdp_params, fsdp_params, atol=atol, rtol=rtol)
            deep_compare(ddp_params, fsdp_params, atol=atol, rtol=rtol)


def get_stats(trainer: Trainer) -> dict[str, np.ndarray]:
    logger = trainer.logger.destinations[0]
    stats = {
        'loss_array':
            logger.get_timeseries('loss/train/total')['loss/train/total'],  # type: ignore
        'accuracy_array':
            logger.get_timeseries('metrics/train/MulticlassAccuracy')  # type: ignore
            ['metrics/train/MulticlassAccuracy'],
    }
    return stats


@pytest.mark.gpu
@world_size(4)
@pytest.mark.parametrize('replication', [2])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_forwards_backwards_correctness(world_size: int, replication: int):
    """Test that training with DDP, FSDP, TP-FSDP results in the same:
        - initial weights
        - forward pass
        - gradients
        - updated weights
    after training for a single step via manually doing forward, backward pass.
    """

    # Initialize trainers with DDP, FSDP, TP-FSDP
    ddp_trainer = get_trainer('ddp', replication=replication)
    fsdp_trainer = get_trainer('fsdp', replication=replication)
    tp_fsdp_trainer = get_trainer('tp-fsdp', replication=replication)

    # Ensure initial model weights are the same
    compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer)

    # Forward pass
    ddp_out = forward_pass(ddp_trainer)
    fsdp_out = forward_pass(fsdp_trainer)
    tp_fsdp_out = forward_pass(tp_fsdp_trainer)

    # Ensure output of the forward pass is the same
    deep_compare(ddp_out, fsdp_out)
    deep_compare(ddp_out, tp_fsdp_out)
    deep_compare(fsdp_out, tp_fsdp_out)

    # Compute gradients
    torch.sum(ddp_out).backward()
    torch.sum(fsdp_out).backward()
    torch.sum(tp_fsdp_out).backward()

    # Ensure the model gradients are the same
    compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer, check_grad=True)

    # Update the model weights
    ddp_trainer.state.optimizers[0].step()
    fsdp_trainer.state.optimizers[0].step()
    tp_fsdp_trainer.state.optimizers[0].step()

    # Ensure the updated model weights are the same
    compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.parametrize('replication', [2])
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fit_correctness(world_size: int, batch_size: int, replication: int):
    """Test that training with DDP, FSDP, TP-FSDP results in the same:
        - updated weights
        - loss
        - accuracy
    after training for multiple steps via trainer.fit().
    """

    # Initialize number of samples in the dataset
    # train_steps = 20  # number of steps to train for
    train_steps = 20
    samples_per_batch = world_size * batch_size // replication
    dataset_size = samples_per_batch * train_steps

    # DDP fit
    ddp_trainer = get_trainer('ddp', size=dataset_size, batch_size=batch_size, replication=replication)
    ddp_trainer.fit()
    ddp_trainer.close()
    ddp_stats = get_stats(ddp_trainer)

    # FSDP fit
    fsdp_trainer = get_trainer('fsdp', size=dataset_size, batch_size=batch_size, replication=replication)
    fsdp_trainer.fit()
    fsdp_trainer.close()
    fsdp_stats = get_stats(fsdp_trainer)

    # TP-FSDP fit
    tp_fsdp_trainer = get_trainer('tp-fsdp', size=dataset_size, batch_size=batch_size, replication=replication)
    tp_fsdp_trainer.fit()
    tp_fsdp_trainer.close()
    tp_fsdp_stats = get_stats(tp_fsdp_trainer)

    # Ensure the updated models weights are the same
    # Drop tolerance due to precision issues across different parallelism strategies
    compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer, atol=1e-5, rtol=1e-3)

    # Compare the loss, accuracy stats
    # Drop tolerance due to precision issues across different parallelism strategies
    deep_compare(ddp_stats, fsdp_stats, atol=6e-5)
    deep_compare(tp_fsdp_stats, fsdp_stats, atol=6e-5)
    deep_compare(ddp_stats, tp_fsdp_stats, atol=6e-5)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.parametrize('tensor_parallel_degree', [1, 2])
def test_tp_train(world_size: int, tensor_parallel_degree: int):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    # For TP to produce the correct result, each TP rank receives the same data
    # In this test, TP ranks receive different data as we are testing the TP
    # mechanism, not actual TP correctness.
    model = SimpleModel()
    optimizer = DecoupledSGDW(model.parameters(), lr=0.1)
    dataset = RandomClassificationDataset(size=8)
    dataloader = DataLoader(dataset, batch_size=2, sampler=dist.get_sampler(dataset))

    layer_plan = {
        'fc1': ColwiseParallel(),
        'fc2': RowwiseParallel(),
    }

    if tensor_parallel_degree == 1:
        expected_warning = 'Received tensor_parallel_degree of 1, which is a no-op. Tensor parallelism will not be used.'
        ctx = pytest.warns(UserWarning, match=expected_warning)
    else:
        ctx = contextlib.nullcontext()

    with ctx:
        trainer = Trainer(
            model=model,
            optimizers=optimizer,
            train_dataloader=dataloader,
            parallelism_config={
                'tp': {
                    'layer_plan': layer_plan,
                    'tensor_parallel_degree': tensor_parallel_degree,
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

    # For TP to produce the correct result, each TP rank receives the same data
    # In this test, TP ranks receive different data as we are testing the TP
    # mechanism, not actual TP correctness.
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


@world_size(4)
@pytest.mark.gpu
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_with_subset_of_params(world_size: int):
    from torch.distributed.tensor.parallel import ColwiseParallel

    # For TP to produce the correct result, each TP rank receives the same data
    # In this test, TP ranks receive different data as we are testing the TP
    # mechanism, not actual TP correctness.
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


@world_size(4)
@pytest.mark.gpu
@pytest.mark.skip('This is broken due to https://github.com/pytorch/pytorch/issues/134095/.')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fsdp_state_dict(world_size: int):
    tp_fsdp_trainer = get_trainer('tp_fsdp', replication=2)
    tp_fsdp_state_dict1 = tp_fsdp_trainer.state.state_dict()  # work sometimes, fails sometimes
    with FSDP.summon_full_params(tp_fsdp_trainer.state.model, with_grads=True):
        tp_fsdp_state_dict2 = tp_fsdp_trainer.state.state_dict()  # fails always

        deep_compare(tp_fsdp_state_dict1['model'], tp_fsdp_state_dict2['model'])
