# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import pytest
import torch
from packaging import version

from icecream import ic
from composer.core.state import fsdp_get_optim_state_dict, fsdp_state_dict_type_context
from composer.utils import reproducibility, FSDPConfig, TPConfig, dist

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from tests.common import (
    RandomClassificationDataset,
    SimpleModel,
    SimpleComposerMLP,
    world_size,
)
from tests.trainer.test_fsdp_checkpoint import _compare_model_params_between_state_dicts
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader


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


def get_trainer(parallelism_config):
    """Trainer for a simple model with any parallelism_config."""
    num_features, num_classes, batch_size, size, seed = 64, 3, 8, 32, 42
    reproducibility.seed_all(seed)

    dataset = RandomClassificationDataset(shape=(num_features,), num_classes=num_classes, size=size, device='cuda') # X=(num_features,), y=(,), i.e. scalar
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


def _forward(trainer):
    batch = next(iter(trainer.state.train_dataloader))
    output = trainer.state.model.forward(batch)
    return output


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_forward(world_size: int):
    """Test that the forward pass with DDP, FSDP, FSDP + TP all match."""
    import warnings
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from icecream import install
    install()
    warnings.filterwarnings("ignore")

    # DDP
    trainer_ddp = get_trainer(parallelism_config=None)

    # FSDP
    fsdp_config = FSDPConfig(state_dict_type='full', sharding_strategy='SHARD_GRAD_OP') # data_parallel_shard_degree=2)
    trainer_fsdp = get_trainer(parallelism_config={'fsdp': fsdp_config})

    # FSDP + TP
    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
    parallelism_config = {'fsdp': fsdp_config, 'tp': tp_config}
    trainer_fsdp_tp = get_trainer(parallelism_config=parallelism_config)

    out_ddp = _forward(trainer_ddp)
    out_fsdp = _forward(trainer_fsdp)
    ic(out_ddp.shape)
    ic(out_fsdp.shape)


    # if dist.get_global_rank() == 0:
    #     with trainer_fsdp.state.model.module.summon_full_params(trainer_fsdp.state.model.module):
    #         with trainer_fsdp_tp.state.model.module.summon_full_params(trainer_fsdp_tp.state.model.module):

    #             # out_ddp = _forward(trainer_ddp)
    #             # out_fsdp = _forward(trainer_fsdp)
    #             # out_fsdp_tp = _forward(trainer_fsdp_tp)

    #             ic(trainer_fsdp.state.state_dict()['model'].keys())
    #             ic(trainer_ddp.state.state_dict()['model'].keys())

    #             ic(trainer_fsdp.state.state_dict()['model'])
    #             ic(trainer_ddp.state.state_dict()['model'])
    #             assert trainer_fsdp.state.state_dict()['model'] == trainer_ddp.state.state_dict()['model']

    # # torch.testing.assert_close(param_ddp, param_fsdp)
    # assert out_ddp.shape == out_fsdp.shape == out_fsdp_tp.shape, f"Outputs have different shapes: {out_ddp.shape=}, {out_fsdp.shape=}, {out_fsdp_tp.shape=}"
    # assert torch.allclose(out_ddp, out_fsdp), f"Outputs have different values: {out_ddp=} and {out_fsdp=}"
    # assert torch.allclose(out_ddp, out_fsdp_tp), f"Outputs have different values: {out_ddp=} and {out_fsdp_tp=}"


# from https://github.com/mosaicml/composer/blob/ce0bffe0bcbfbf290d1a670c465c870806138bcd/tests/trainer/test_fsdp_checkpoint.py#L1036
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


# @pytest.mark.gpu
# @world_size(4)
# @pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
# @pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
# def test_tp_init_params(world_size: int):
#     """Test that models with DDP, FSDP, FSDP + TP all have the same weights after initilization."""
#     from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

#     # DDP
#     trainer_ddp = get_trainer(parallelism_config=None)
#     state_ddp = trainer_ddp.state

#     # FSDP
#     fsdp_config = FSDPConfig(state_dict_type='sharded') # {'data_parallel_shard_degree': 2}
#     trainer_fsdp = get_trainer(parallelism_config={'fsdp': fsdp_config})
#     state_dict_fsdp = trainer_ddp.state.state_dict()
#     ic(type(state_dict_fsdp), state_dict_fsdp)

#     # state_dict_fsdp = get_mono_state_dict_from_sharded_one(trainer_fsdp)

#     parameters = trainer_fsdp.state.model.named_parameters()
#     [ic(name, parameter.shape) for name, parameter in parameters]
#     with trainer_fsdp.state.model.module.summon_full_params(trainer_fsdp.state.model.module):  # type: ignore
#         parameters = ((name, parameter.clone()) for name, parameter in parameters)
#         [ic(name, parameter.shape) for name, parameter in parameters]
#     [ic(name, parameter.shape) for name, parameter in parameters]

#     # FSDP + TP
#     layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
#     tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
#     parallelism_config = {'fsdp': fsdp_config, 'tp': tp_config}
#     trainer_fsdp_tp = get_trainer(parallelism_config=parallelism_config)
#     state_dict_fsdp_tp = get_mono_state_dict_from_sharded_one(trainer_fsdp_tp)

#     # from https://github.com/mosaicml/composer/blob/ce0bffe0bcbfbf290d1a670c465c870806138bcd/tests/trainer/test_fsdp_checkpoint.py#L1057
#     # We are comparing full state dicts (all optim and model parameters are gathered on only rank 0)
#     # so we only need to compare on rank 0. Comparing on other ranks may cause errors because some state_dicts will be empty.
#     if dist.get_global_rank() == 0:
#         _compare_model_params_between_state_dicts(state_ddp, state_dict_fsdp)
#         with pytest.raises(Exception):
#             _compare_model_params_between_state_dicts(state_ddp, state_dict_fsdp_tp)


@pytest.mark.gpu
@world_size(4)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.3'), reason='Requires PyTorch 2.3+')
@pytest.mark.filterwarnings(r'ignore:.*\(TP\) is experimental.*:FutureWarning')
def test_tp_init_params(world_size: int):
    """Test that models with DDP, FSDP, FSDP + TP all have the same weights after initilization."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._tensor import DTensor

    # DDP
    trainer_ddp = get_trainer(parallelism_config=None)
    module_ddp = trainer_ddp.state.model.module
    ic(module_ddp)
    params_ddp = {name: param for name, param in module_ddp.named_parameters()}

    # FSDP
    fsdp_config = FSDPConfig(state_dict_type='full')
    trainer_fsdp = get_trainer(parallelism_config={'fsdp': fsdp_config})
    module_fsdp = trainer_fsdp.state.model.module
    ic(module_fsdp)
    with module_fsdp.summon_full_params(module_fsdp):
        params_fsdp = {name: param.clone() for name, param in module_fsdp.named_parameters()}

    # FSDP + TP
    layer_plan = {'fc1': ColwiseParallel(), 'fc2': RowwiseParallel()}
    tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
    parallelism_config = {'fsdp': fsdp_config, 'tp': tp_config}
    trainer_fsdp_tp = get_trainer(parallelism_config=parallelism_config)
    module_fsdp_tp = trainer_fsdp_tp.state.model.module
    ic(module_fsdp_tp)
    with module_fsdp_tp.summon_full_params(module_fsdp_tp):
        params_fsdp_tp = {name: param.clone() for name, param in module_fsdp_tp.named_parameters()}

    def _cast(param):
        if isinstance(param, ShardedTensor): param = param.local_tensor()
        if isinstance(param, DTensor): param = param.to_local()
        return param

    def _chunk(param, name):

        rank = dist.get_local_rank()
        world_size = dist.get_world_size()
        tp_degree = tp_config.tensor_parallel_degree
        denominator = world_size * tp_degree
        ic(rank, world_size, tp_degree, denominator)

        layer_plan = {'0.weight' if layer == 'fc1' else '2.weight': plan for layer, plan in copy.deepcopy(tp_config.layer_plan).items()}
        plan = layer_plan[name]
        if isinstance(plan, RowwiseParallel):
            dim = param.shape[1]
            start_idx, end_idx = int(rank * (dim / tp_degree)), int((rank + 1) * (dim / tp_degree))
            ic('RowwiseParallel', start_idx, end_idx)
            return param[:, start_idx: end_idx]
        elif isinstance(layer_plan[name], ColwiseParallel):
            dim = param.shape[0]
            start_idx, end_idx = int(rank * (dim / tp_degree)), int((rank + 1) * (dim / tp_degree))
            ic('ColwiseParallel', start_idx, end_idx)
            return param[start_idx: end_idx, :]
        else:
            return None

    ic('Params DDP:')
    [ic(name, param.shape, _cast(param).shape) for name, param in params_ddp.items()]
    ic('Params FSDP:')
    [ic(name, param.shape, _cast(param).shape) for name, param in params_fsdp.items()]
    ic('Params FSDP + TP:')
    [ic(name, param.shape, _cast(param).shape) for name, param in params_fsdp_tp.items()]

    # compare weights of ddp, fsdp, fsdp-tp
    for (name_ddp, param_ddp), (name_fsdp, param_fsdp), (name_fsdp_tp, param_fsdp_tp) in zip(params_ddp.items(), params_fsdp.items(), params_fsdp_tp.items()):
        param_ddp = _cast(param_ddp)
        param_fsdp = _cast(param_fsdp)
        param_fsdp_tp = _cast(param_fsdp_tp)

        ic(param_ddp.shape, param_fsdp.shape, param_fsdp_tp.shape)
        param_ddp = _chunk(param_ddp, name_ddp.lstrip('module.'))
        param_fsdp = _chunk(param_fsdp, name_fsdp)
        ic(param_ddp.shape, param_fsdp.shape, param_fsdp_tp.shape)

        torch.testing.assert_close(param_ddp, param_fsdp)
        torch.testing.assert_close(param_ddp, param_fsdp_tp)
