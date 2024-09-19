# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Any, Optional, Sequence, TypeVar

E = TypeVar('E', bound=BaseException)

import numpy as np
import pytest
import torch
from icecream import ic
from packaging import version
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.utils.data import DataLoader, Dataset

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from composer.utils import FSDPConfig, ParallelismConfig, TPConfig, dist, reproducibility
from tests.common import (
    RandomClassificationDataset,
    SimpleComposerMLP,
    SimpleModel,
    world_size,
)


class RandomClassificationDatasetReplicated(Dataset):
    """Like RandomClassificationDataset but samples are replicated across TP groups.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(
        self,
        shape: Sequence[int] = (1, 1, 1),
        size: int = 100,
        num_classes: int = 2,
        device: Optional[torch.device] = None,
        seed: int = 44,
        replication: int = 2,
    ):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.device = device
        self.rank = dist.get_local_rank()
        self.world_size = dist.get_world_size()
        self.n_tp_groups = replication  # the number of tp groups that we are replicating across
        self.seed = seed
        self.x: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

    def _generate_data(self):
        tp_group_id = self.rank // self.n_tp_groups
        seed = self.seed + tp_group_id  # all ranks in the same TP group have the same seed
        reproducibility.seed_all(seed)
        self.x = torch.randn(self.size, *self.shape, device=self.device)
        self.y = torch.randint(0, self.num_classes, size=(self.size,), device=self.device)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.x is None and self.y is None:
            self._generate_data()

        assert self.x is not None
        assert self.y is not None

        rank_idx = idx // self.world_size
        return self.x[rank_idx], self.y[rank_idx]


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
    num_features: int = 2,
    seed: int = 44,
    device: torch.device = torch.device('cuda'),
    replication: int = 0,
):
    """Trainer for a simple model with any parallelism_config."""

    reproducibility.seed_all(seed)

    if replication:
        dataset: Dataset = RandomClassificationDatasetReplicated(
            shape=(num_features,),
            num_classes=num_classes,
            size=size,
            device=device,
            replication=replication,
        )  # X=(num_features,), y=(,), i.e. scalar
    else:
        dataset: Dataset = RandomClassificationDataset(
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
    seed: int = 44,
    device: torch.device = torch.device('cuda'),
    replication: int = 0,
):
    ddp_trainer = get_trainer(
        size=size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_features=num_features,
        seed=seed,
        device=device,
        replication=replication,
    )
    return ddp_trainer


def get_fsdp_trainer(
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 44,
    device: torch.device = torch.device('cuda'),
    replication: int = 0,
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
        replication=replication,
    )
    return fsdp_trainer


def get_tp_fsdp_trainer(
    size: int = 4,
    batch_size: int = 1,
    num_classes: int = 2,
    num_features: int = 2,
    seed: int = 44,
    device: torch.device = torch.device('cuda'),
    replication: int = 0,
    tensor_parallel_degree: int = 2,
):
    fsdp_config = FSDPConfig(
        state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        mixed_precision='full',
        use_orig_params=True,
    )

    if replication:
        layer_plan = {
            'fc1': ColwiseParallel(),
            'fc2': RowwiseParallel(),
        }
        assert tensor_parallel_degree == replication
    else:
        layer_plan = {
            'fc1': GatherColwiseParallel(),
            'fc2': RowwiseParallel(output_layouts=Shard(0)),
        }

    tp_config = TPConfig(
        layer_plan=layer_plan,
        tensor_parallel_degree=tensor_parallel_degree,
    )
    parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)

    tp_fsdp_trainer = get_trainer(
        parallelism_config=parallelism_config,
        size=size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_features=num_features,
        seed=seed,
        device=device,
        replication=replication,
    )
    return tp_fsdp_trainer


def forward_pass(trainer):
    reproducibility.seed_all(42)
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


def compare_modules(
    module1: dict[str, Any],
    module2: dict[str, Any],
    check_grad: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None
):
    module_type = 'Gradients' if check_grad else 'Parameters'

    for (param1_name, param1), (param2_name, param2) in zip(module1.items(), module2.items()):

        assert param1_name == param2_name

        if check_grad:
            param1 = param1.grad
            param2 = param2.grad

        if isinstance(param1, DTensor):
            param1 = param1.redistribute(device_mesh=param1.device_mesh, placements=[Replicate()]).to_local()
        if isinstance(param2, DTensor):
            param2 = param2.redistribute(device_mesh=param2.device_mesh, placements=[Replicate()]).to_local()

        torch.testing.assert_close(
            param1,
            param2,
            atol=atol,
            rtol=rtol,
            msg=f'{module_type} are not close enough:\n{param1=}\n{param2=}',
        )


def compare_models(
    ddp_trainer: Trainer,
    fsdp_trainer: Trainer,
    tp_fsdp_trainer: Trainer,
    check_grad: bool = False,
    atol: Optional[float] = None,
    rtol: Optional[float] = None
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

            ic(ddp_params, fsdp_params, tp_fsdp_params)

            compare_modules(ddp_params, fsdp_params, check_grad=check_grad, atol=atol, rtol=rtol)
            compare_modules(tp_fsdp_params, fsdp_params, check_grad=check_grad, atol=atol, rtol=rtol)
            compare_modules(ddp_params, fsdp_params, check_grad=check_grad, atol=atol, rtol=rtol)


@contextmanager
def fail_without_replication(replication: int, exception: type[E], error_error_msg: str):
    if replication:
        yield
    else:
        with pytest.raises(exception, match=error_error_msg):
            yield


def get_stats(trainer: Trainer) -> dict[str, np.ndarray]:
    logger = trainer.logger.destinations[0]
    stats = {
        'loss_array': logger.get_timeseries('loss/train/total')['loss/train/total'],
        'accuracy_array': logger.get_timeseries('metrics/train/MulticlassAccuracy')['metrics/train/MulticlassAccuracy'],
    }
    return stats


@pytest.mark.gpu
@world_size(4)
@pytest.mark.parametrize('replication', [0, 2])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_forwards_backwards_correctness(world_size: int, replication: int):
    """Test that training with DDP, FSDP, TP-FSDP results in the same:
        - initial weights
        - forward pass
        - gradients
        - updated weights
    for a single step.
    """

    # Initialize trainers with DDP, FSDP, TP-FSDP
    ddp_trainer = get_ddp_trainer(replication=replication)
    fsdp_trainer = get_fsdp_trainer(replication=replication)
    tp_fsdp_trainer = get_tp_fsdp_trainer(replication=replication)

    # Ensure initial model weights are the same
    compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer)

    # Forward pass
    ddp_out = forward_pass(ddp_trainer)
    fsdp_out = forward_pass(fsdp_trainer)
    tp_fsdp_out = forward_pass(tp_fsdp_trainer)

    # Ensure output of the forward pass is the same
    with FSDP.summon_full_params(fsdp_trainer.state.model):
        with FSDP.summon_full_params(tp_fsdp_trainer.state.model):
            compare_modules({'': ddp_out}, {'': fsdp_out})
            compare_modules({'': ddp_out}, {'': tp_fsdp_out})
            compare_modules({'': fsdp_out}, {'': tp_fsdp_out})

    # Compute gradients
    torch.sum(ddp_out).backward()
    torch.sum(fsdp_out).backward()
    torch.sum(tp_fsdp_out).backward()

    # Ensure the gradients are the same
    # We expect this test to fail without replication, i.e. when replication=0
    error_error_msg = 'Gradients are not close enough:*'
    with fail_without_replication(replication, AssertionError, error_error_msg):
        compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer, check_grad=True)

    # Update the model weights
    ddp_trainer.state.optimizers[0].step()
    fsdp_trainer.state.optimizers[0].step()
    tp_fsdp_trainer.state.optimizers[0].step()

    # Ensure the updated weights are the same
    # We expect this test to fail without replication, i.e. when replication=0
    error_error_msg = 'Parameters are not close enough:*'
    with fail_without_replication(replication, AssertionError, error_error_msg):
        compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.parametrize('replication', [0, 2])
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

    # Initialize
    train_steps = 20  # number of steps to train for
    n_samples = world_size * batch_size
    dataset_size = n_samples * train_steps

    # DDP fit
    ddp_trainer = get_ddp_trainer(size=dataset_size, batch_size=batch_size, replication=replication)
    ddp_trainer.fit()
    ddp_trainer.close()
    ddp_stats = get_stats(ddp_trainer)

    # FSDP fit
    fsdp_trainer = get_fsdp_trainer(size=dataset_size, batch_size=batch_size, replication=replication)
    fsdp_trainer.fit()
    fsdp_trainer.close()
    fsdp_stats = get_stats(fsdp_trainer)

    # TP-FSDP fit
    tp_fsdp_trainer = get_tp_fsdp_trainer(size=dataset_size, batch_size=batch_size, replication=replication)
    tp_fsdp_trainer.fit()
    tp_fsdp_trainer.close()
    tp_fsdp_stats = get_stats(tp_fsdp_trainer)

    ic(ddp_stats, fsdp_stats, tp_fsdp_stats)

    # Ensure the updated models weights are the same
    # Drop tolerance due to precision issues across different parallelism strategies
    # We expect this test to fail without replication, i.e. when replication=0
    error_error_msg = 'Parameters are not close enough:*'
    with fail_without_replication(replication, AssertionError, error_error_msg):
        compare_models(ddp_trainer, fsdp_trainer, tp_fsdp_trainer, atol=1e-5, rtol=1e-3)

        # Compare loss between DDP, FSDP, TP-FSDP
        np.testing.assert_allclose(
            ddp_stats['loss_array'],
            fsdp_stats['loss_array'],
            atol=6e-5,
            err_msg='Loss arrays of DDP and FSDP are not close enough.',
        )
        np.testing.assert_allclose(
            ddp_stats['loss_array'],
            tp_fsdp_stats['loss_array'],
            atol=6e-5,
            err_msg='Loss arrays of DDP and TP-FSDP are not close enough.',
        )
        np.testing.assert_allclose(
            fsdp_stats['loss_array'],
            tp_fsdp_stats['loss_array'],
            atol=6e-5,
            err_msg='Loss arrays of FSDP and TP-FSDP are not close enough.',
        )

        # Compare accuracy between DDP, FSDP, TP-FSDP
        np.testing.assert_allclose(
            ddp_stats['accuracy_array'],
            fsdp_stats['accuracy_array'],
            err_msg='Accuracy arrays of DDP and FSDP are not close enough',
        )
        np.testing.assert_allclose(
            ddp_stats['accuracy_array'],
            tp_fsdp_stats['accuracy_array'],
            err_msg='Accuracy arrays of DDP and FSDP-TP are not close enough',
        )
        np.testing.assert_allclose(
            fsdp_stats['accuracy_array'],
            tp_fsdp_stats['accuracy_array'],
            err_msg='Accuracy arrays of FSDP and FSDP-TP are not close enough',
        )


@world_size(4)
@pytest.mark.gpu
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fsdp_trainer_2(world_size: int):
    # from icecream import ic

    ###############
    # Parameters
    ###############

    size: int = 4
    batch_size: int = 1
    num_classes: int = 2
    num_features: int = 2
    seed: int = 44
    tensor_parallel_degree: int = 2
    device: torch.device = torch.device('cuda')
    output_dir: str = '/my-tmp/'

    reproducibility.seed_all(seed)
    rank = dist.get_local_rank()

    ###############
    # DataLoader
    ###############

    my_dataset = MyDataset(
        shape=(num_features,),
        num_classes=num_classes,
        size=size,
        device=device,
        rank=rank,
    )  # X=(num_features,), y=(,), i.e. scalar

    # for i in range(len(my_dataset)):
    #     x, y = my_dataset[i]
    #     ic(rank)
    #     ic(x.shape, x)
    #     ic(y.shape, y)
    #     ic('\n')

    dataloader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        sampler=dist.get_sampler(my_dataset),
    )

    # pytorch_dataset = RandomClassificationDataset(
    #     shape=(num_features,),
    #     num_classes=num_classes,
    #     size=size,
    #     device=device,
    # )

    # # clean directory
    # rmtree(output_dir)

    # # columns = {'x': 'ndarray:float32:2', 'y': 'int64'} # 2 -> features
    # columns = {'x': 'pkl', 'y': 'int64'}
    # with MDSWriter(out=output_dir, columns=columns) as out:
    #     for i in range(len(pytorch_dataset)):
    #         x, y = pytorch_dataset[i]
    #         out.write({'x': x.cpu().detach().numpy(), 'y': y.cpu().detach().numpy()})
    #         # out.write({'x': x.numpy(), 'y': y.numpy()})

    # streaming_dataset = StreamingDataset(
    #     local=output_dir,
    #     replication=tensor_parallel_degree,
    #     batch_size=batch_size,
    #     allow_unsafe_types=True
    # )

    # dataloader = DataLoader(
    #     streaming_dataset,
    # )

    ###############
    # Model
    ###############

    model = SimpleComposerMLP(
        num_features=num_features,
        device=device,
        num_classes=num_classes,
    )

    #####################
    # Parallelism Config
    #####################

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
        tensor_parallel_degree=tensor_parallel_degree,
    )
    parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)

    #####################
    # Trainer
    #####################

    tp_fsdp_trainer = Trainer(
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

    tp_fsdp_trainer.fit()


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


@world_size(4)
@pytest.mark.gpu
@pytest.mark.skip('This is broken due to https://github.com/pytorch/pytorch/issues/134095/.')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_fsdp_state_dict(world_size: int):
    tp_fsdp_trainer = get_tp_fsdp_trainer(replication=2)
    tp_fsdp_state_dict1 = tp_fsdp_trainer.state.state_dict()  # work sometimes, fails sometimes
    with FSDP.summon_full_params(tp_fsdp_trainer.state.model, with_grads=True):
        tp_fsdp_state_dict2 = tp_fsdp_trainer.state.state_dict()  # fails always
