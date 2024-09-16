# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import pytest
import torch
from packaging import version
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from composer.callbacks import MemoryMonitor
from composer.core.state import fsdp_state_dict_type_context
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from composer.utils import FSDPConfig, ParallelismConfig, TPConfig, dist, reproducibility
from tests.common import (
    RandomClassificationDataset,
    SimpleComposerMLP,
    SimpleModel,
    world_size,
)
from tests.trainer.test_fsdp_checkpoint import get_mono_state_dict_from_sharded_one

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
        use_local_output: bool = True,
    ):
        super().__init__()
        # Inputs over the TP dimension are sharded by device batches.
        self.input_layouts = (Shard(0),)
        # All-gather inputs so that each GPU now has the same input activations.
        self.desired_input_layouts = (Replicate(),)
        # self.output_layouts = (Shard(-1), )
        self.use_local_output = use_local_output


def get_trainer(
    parallelism_config: Optional[ParallelismConfig] = None,
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 6,
    seed: int = 42,
    device: torch.device = 'cuda',
):
    """Trainer for a simple model with any parallelism_config."""

    reproducibility.seed_all(seed)
    dataset = RandomClassificationDataset(
        shape=(num_features,),
        num_classes=num_classes,
        size=size,
        device=device,
    )  # X=(num_features,), y=(,), i.e. scalar
    dataloader = DataLoader(
        dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=batch_size,
    )  # X=(batch_size, num_features), y=(batch_size,)
    model = SimpleComposerMLP(num_features=num_features, device=device, num_classes=num_classes)

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


def get_ddp_trainer(
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 42,
    device: torch.device = 'cuda',
):
    ddp_trainer = get_trainer(
        size=size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_features=num_features,
        seed=seed,
        device=device,
    )
    return ddp_trainer


def get_fsdp_trainer(
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 42,
    device: torch.device = 'cuda',
):
    fsdp_config = FSDPConfig(
        state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        mixed_precision='full',
        use_orig_params=True,
    )
    parallelism_config = ParallelismConfig(fsdp=fsdp_config)

    fsdp_trainer = get_trainer(
        parallelism_config=parallelism_config,
        size=size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_features=num_features,
        seed=seed,
        device=device,
    )
    return fsdp_trainer


def get_tp_fsdp_trainer(
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 42,
    device: torch.device = 'cuda',
):
    fsdp_config = FSDPConfig(
        state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        mixed_precision='full',
        use_orig_params=True,
    )
    layer_plan = {
        'fc1': GatherColwiseParallel(),
        'fc2': RowwiseParallel(output_layouts=Shard(0)),
    }
    tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
    parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)

    fsdp_trainer = get_trainer(
        parallelism_config=parallelism_config,
        size=size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_features=num_features,
        seed=seed,
        device=device,
    )
    return fsdp_trainer


def forward_pass(trainer):
    batch = next(iter(trainer.state.train_dataloader))
    output = trainer.state.model.forward(batch)
    return output


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_forward(world_size: int):
    """Test that DDP, FSDP, TP-FSDP do the same forward pass."""

    # DDP forward pass
    ddp_trainer = get_ddp_trainer()
    ddp_out = forward_pass(ddp_trainer)

    # FSDP forward pass
    fsdp_trainer = get_fsdp_trainer()
    fsdp_out = forward_pass(fsdp_trainer)

    # TP-FSDP forward pass
    tp_fsdp_trainer = get_tp_fsdp_trainer()
    tp_fsdp_out = forward_pass(tp_fsdp_trainer)

    assert ddp_out.shape == fsdp_out.shape == tp_fsdp_out.shape, f'Outputs have different shapes: {ddp_out.shape=}, {fsdp_out.shape=}, {tp_fsdp_out.shape=}'
    assert torch.allclose(ddp_out, fsdp_out, atol=1e-3), f'Outputs have different values: {ddp_out=} and {fsdp_out=}'
    assert torch.allclose(
        ddp_out,
        tp_fsdp_out,
        atol=1e-3,
    ), f'Outputs have different values: {ddp_out=} and {tp_fsdp_out=}'


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_hang(world_size: int):

    fsdp_trainer = get_fsdp_trainer()
    fsdp_trainer.fit()
    print('fsdp_state_dict_1')
    fsdp_state_dict_1 = fsdp_trainer.state.state_dict()
    print(fsdp_state_dict_1)
    fsdp_trainer.close()
    print('fsdp_state_dict_2')
    fsdp_state_dict_2 = fsdp_trainer.state.state_dict()
    print(fsdp_state_dict_2)

    with fsdp_state_dict_type_context(fsdp_trainer.state.model, state_dict_type='full'):
        print('fsdp_state_dict_3')
        fsdp_state_dict_3 = fsdp_trainer.state.model.state_dict()
        print(fsdp_state_dict_3)
        print('fsdp_state_dict_4')
        fsdp_state_dict_4 = fsdp_trainer.state.state_dict()
        print(fsdp_state_dict_4)

    # if dist.get_local_rank() == 0:
    #     # fsdp_state_dict_6 HANGS!!
    #     print('fsdp_state_dict_5')
    #     with fsdp_state_dict_type_context(fsdp_trainer.state.model, state_dict_type='full'):
    #         fsdp_state_dict_5 = fsdp_trainer.state.state_dict()
    #         print(fsdp_state_dict_5)

    #     # fsdp_state_dict_6 HANGS!!
    #     print('fsdp_state_dict_6')
    #     with fsdp_state_dict_type_context(fsdp_trainer.state.model, state_dict_type='full'):
    #         fsdp_state_dict_6 = fsdp_trainer.state.model.state_dict()
    #         print(fsdp_state_dict_6)

    tp_fsdp_trainer = get_tp_fsdp_trainer()
    tp_fsdp_trainer.fit()
    print('tp_fsdp_state_dict_1')
    tp_fsdp_state_dict_1 = tp_fsdp_trainer.state.state_dict()
    print(tp_fsdp_state_dict_1)
    tp_fsdp_trainer.close()
    print('tp_fsdp_state_dict_2')
    tp_fsdp_state_dict_2 = tp_fsdp_trainer.state.state_dict()
    print(tp_fsdp_state_dict_2)

    with fsdp_state_dict_type_context(tp_fsdp_trainer.state.model, state_dict_type='full'):
        print('tp_fsdp_state_dict_3')
        tp_fsdp_state_dict_3 = tp_fsdp_trainer.state.model.state_dict()
        print(tp_fsdp_state_dict_3)
        print('tp_fsdp_state_dict_4')
        tp_fsdp_state_dict_4 = tp_fsdp_trainer.state.state_dict()
        print(tp_fsdp_state_dict_4)

    # if dist.get_local_rank() == 0:
    #     # tp_fsdp_state_dict_5 HANGS!!
    #     print('tp_fsdp_state_dict_5')
    #     with fsdp_state_dict_type_context(tp_fsdp_trainer.state.model, state_dict_type='full'):
    #         tp_fsdp_state_dict_5 = tp_fsdp_trainer.state.state_dict()
    #         print(tp_fsdp_state_dict_5)

    #     # tp_fsdp_state_dict_6 HANGS!!
    #     print('tp_fsdp_state_dict_6')
    #     with fsdp_state_dict_type_context(tp_fsdp_trainer.state.model, state_dict_type='full'):
    #         tp_fsdp_state_dict_6 = tp_fsdp_trainer.state.model.state_dict()
    #         print(tp_fsdp_state_dict_6)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_gradients(world_size: int):
    """Test that DDP, FSDP, TP-FSDP output the same gradients."""
    # from icecream import ic

    # DDP gradients
    ddp_trainer = get_ddp_trainer()
    ddp_out = forward_pass(ddp_trainer)
    torch.sum(ddp_out).backward()
    ddp_params = {name: param for name, param in ddp_trainer.state.model.named_parameters()}
    ddp_trainer.close()

    # FSDP gradients
    fsdp_trainer = get_fsdp_trainer()
    fsdp_out = forward_pass(fsdp_trainer)
    torch.sum(fsdp_out).backward()
    # with FSDP.summon_full_params(fsdp_trainer.state.model, with_grads=True):
    fsdp_params = {name: param for name, param in fsdp_trainer.state.model.named_parameters()}
    fsdp_trainer.close()

    # TP-FSDP gradients
    tp_fsdp_trainer = get_tp_fsdp_trainer()
    tp_fsdp_out = forward_pass(tp_fsdp_trainer)
    torch.sum(tp_fsdp_out).backward()
    # with FSDP.summon_full_params(tp_fsdp_trainer.state.model, with_grads=True):
    tp_fsdp_params = {name: param for name, param in tp_fsdp_trainer.state.model.named_parameters()}
    tp_fsdp_trainer.close()

    rank = dist.get_local_rank()
    for (ddp_name, ddp_param), (fsdp_name, fsdp_param), (tp_fsdp_name, tp_fsdp_param) in zip(
        ddp_params.items(), fsdp_params.items(), tp_fsdp_params.items()
    ):
        print(f'{rank=}')
        print('\nDDP:\n', ddp_name, ddp_param.shape, ddp_param)
        if ddp_param.grad is not None: print(ddp_param.grad.shape, ddp_param.grad)
        print('\nFSDP:\n', fsdp_name, fsdp_param.shape, fsdp_param)
        if fsdp_param.grad is not None: print(fsdp_param.grad.shape, fsdp_param.grad)
        print('\nTP-FSDP:\n', tp_fsdp_name, tp_fsdp_param.shape, tp_fsdp_param)
        if tp_fsdp_param.grad is not None: print(tp_fsdp_param.grad.shape, tp_fsdp_param.grad)

        # summon_full_params puts all of the parameters on rank 0
        if rank != 0:
            continue

        torch.testing.assert_close(
            ddp_param.grad, fsdp_param.grad,
            msg='DDP and FSDP gradients are not close enough.'
            )
        torch.testing.assert_close(
            ddp_param.grad, tp_fsdp_param.grad,
            msg='DDP and FSDP gradients are not close enough.'
            )
        torch.testing.assert_close(
            fsdp_param.grad, tp_fsdp_param.grad,
            msg='DDP and FSDP gradients are not close enough.'
            )



@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fit_weights(world_size: int):
    """Test that DDP, FSDP, TP-FSDP have the same weights after calling fit, i.e. forward, backward pass."""
    # from icecream import ic

    # DDP gradients
    print('ddp_trainer')
    ddp_trainer = get_ddp_trainer()
    ddp_trainer.fit()
    ddp_state_dict = ddp_trainer.state.state_dict()
    print(f'{ddp_state_dict=}')
    ddp_trainer.close()

    # FSDP gradients
    print('fsdp_trainer')
    fsdp_trainer = get_fsdp_trainer()
    fsdp_trainer.fit()
    fsdp_state_dict = fsdp_trainer.state.state_dict()
    print(f'{fsdp_state_dict=}')
    fsdp_state_dict_2 = get_mono_state_dict_from_sharded_one(fsdp_trainer)
    print(f'{fsdp_state_dict_2=}')
    fsdp_trainer.close()

    # TP-FSDP gradients
    print('tp_fsdp_trainer')
    tp_fsdp_trainer = get_tp_fsdp_trainer()
    tp_fsdp_trainer.fit()
    tp_fsdp_state_dict = tp_fsdp_trainer.state.state_dict()
    print(f'{tp_fsdp_state_dict=}')
    tp_fsdp_state_dict_2 = get_mono_state_dict_from_sharded_one(tp_fsdp_trainer)
    print(f'{tp_fsdp_state_dict_2=}')
    tp_fsdp_trainer.close()

    # for name, param in tp_fsdp_trainer.state.model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.grad.shape, param.grad)

    # if dist.get_local_rank() == 0:
    #     pass

        # todo:
        #! reaname keys, e.g. module.2.weight -> fc2.weight
        #! compare model, optimizer states
        #! use _compare_optims_between_state_dicts, _compare_model_params_between_state_dicts from test_fsdp_checkpoint

        # removes 'module.' from all state dict keys in-place
        # consume_prefix_in_state_dict_if_present(tp_fsdp_state_dict_2['model'], 'module.')
        # print(f'{tp_fsdp_state_dict_2=}')
        # consume_prefix_in_state_dict_if_present(tp_fsdp_state_dict_2['optimizers'], 'module.')
        # print(f'{tp_fsdp_state_dict_2=}')

        # assert fsdp_state_dict_2 == tp_fsdp_state_dict_2


def get_stats(trainer: Trainer) -> dict[str, np.ndarray]:
    logger = trainer.logger.destinations[0]
    stats = {
        'loss_array': logger.get_timeseries('loss/train/total')['loss/train/total'],
        'accuracy_array': logger.get_timeseries('metrics/train/MulticlassAccuracy')['metrics/train/MulticlassAccuracy'],
    }
    return stats


@pytest.mark.gpu
@world_size(4)
@pytest.mark.parametrize('batch_size', [1, 4])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fit(batch_size: int, world_size: int):
    """Test that DDP, FSDP, TP-FSDP have the same trainer.fit(), i.e. output the same loss and accuracy."""

    # Initialize
    train_steps = 20 # number of steps to train for
    dataset_size = world_size * batch_size * train_steps

    # DDP fit
    ddp_trainer = get_ddp_trainer(size=dataset_size, batch_size=batch_size)
    ddp_trainer.fit()
    ddp_trainer.close()
    ddp_stats = get_stats(ddp_trainer)

    # FSDP fit
    fsdp_trainer = get_fsdp_trainer(size=dataset_size, batch_size=batch_size)
    fsdp_trainer.fit()
    fsdp_trainer.close()
    fsdp_stats = get_stats(fsdp_trainer)

    # TP-FSDP fit
    tp_fsdp_trainer = get_tp_fsdp_trainer(size=dataset_size, batch_size=batch_size)
    tp_fsdp_trainer.fit()
    tp_fsdp_trainer.close()
    tp_fsdp_stats = get_stats(tp_fsdp_trainer)

    # Compare loss between DDP, FSDP, TP-FSDP
    np.testing.assert_allclose(
        ddp_stats['loss_array'],
        fsdp_stats['loss_array'],
        atol=5e-2,
        err_msg='Loss arrays of DDP and FSDP are not close enough.',
    )
    np.testing.assert_allclose(
        ddp_stats['loss_array'],
        tp_fsdp_stats['loss_array'],
        atol=5e-2,
        err_msg='Loss arrays of DDP and TP-FSDP are not close enough.',
    )
    np.testing.assert_allclose(
        fsdp_stats['loss_array'],
        tp_fsdp_stats['loss_array'],
        atol=5e-2,
        err_msg='Loss arrays of FSDP and TP-FSDP are not close enough.',
    )

    # Compare accuracy between DDP, FSDP, TP-FSDP
    np.testing.assert_allclose(
        ddp_stats['accuracy_array'],
        fsdp_stats['accuracy_array'],
        atol=0.3,
        err_msg='Accuracy arrays of DDP and FSDP are not close enough',
    )
    np.testing.assert_allclose(
        ddp_stats['accuracy_array'],
        tp_fsdp_stats['accuracy_array'],
        atol=0.3,
        err_msg='Accuracy arrays of DDP and FSDP-TP are not close enough',
    )
    np.testing.assert_allclose(
        fsdp_stats['accuracy_array'],
        tp_fsdp_stats['accuracy_array'],
        atol=0.3,
        err_msg='Accuracy arrays of FSDP and FSDP-TP are not close enough',
    )
