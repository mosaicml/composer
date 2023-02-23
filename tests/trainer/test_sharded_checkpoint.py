# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import textwrap

import numpy as np
import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.markers import world_size


def get_trainer(save_folder=None,
                save_filename='ba{batch}-rank{rank}.pt',
                num_features=2,
                num_classes=2,
                fsdp_state_dict_type='full',
                load_path=None):
    model = SimpleModel(num_features=num_features, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_features, 1, 1), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    trainer = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'min_params': 16,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': 'FULL_SHARD'
        },
        save_folder=save_folder,
        max_duration='2ba',
        save_interval='2ba',
        save_filename=save_filename,
        load_path=load_path,
        progress_bar=False,
        log_to_console=False,
    )
    return trainer


def _compare_optims_between_state_dicts(state_dict1, state_dict2):
    # Check that optim params are equal between checkpoint and in memory optimizer
    state_dict1_optim_params = state_dict1['optimizers']['Adam']['state']
    state_dict2_optim_params = state_dict2['optimizers']['Adam']['state']
    state_dict1_keys = set(state_dict1_optim_params.keys())
    state_dict2_keys = set(state_dict2_optim_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""")

    for param_name in state_dict2_optim_params.keys():
        state_dict1_param_moment_dict = state_dict1_optim_params[param_name]
        state_dict2_param_moment_dict = state_dict2_optim_params[param_name]
        for moment_name in state_dict2_param_moment_dict.keys():
            state_dict1_moment = state_dict1_param_moment_dict[moment_name]
            state_dict2_moment = state_dict2_param_moment_dict[moment_name]
            assert torch.equal(
                state_dict1_moment,
                state_dict2_moment), f'Moment {moment_name} for parameter {param_name} not the same between state dicts'


def _compare_model_params_between_state_dicts(state_dict1, state_dict2):
    # Check that model params are equal between in memory mode and checkpoint
    state_dict1_model_params = state_dict1['model']
    state_dict2_model_params = state_dict2['model']

    state_dict1_keys = set(state_dict1_model_params.keys())
    state_dict2_keys = set(state_dict2_model_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys that belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""")

    for param_name in state_dict2_model_params.keys():
        state_dict1_model_tensor = state_dict1_model_params[param_name]
        state_dict2_model_tensor = state_dict2_model_params[param_name]
        assert torch.equal(state_dict1_model_tensor,
                           state_dict2_model_tensor), f'Weight named {param_name} not the same between state_dicts'


@pytest.mark.gpu
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_save(world_size, tmp_path: pathlib.Path):
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    num_features = 3
    num_classes = 2

    expected_layer_shapes = [(5, num_features), (5,), (num_classes, 5), (num_classes,)]
    layer1_weights_shape, layer1_bias_shape, layer2_weights_shape, layer2_bias_shape = expected_layer_shapes
    expected_total_num_params = sum([np.prod(shape) for shape in expected_layer_shapes])

    trainer = get_trainer(save_folder=str(save_folder),
                          save_filename=save_filename,
                          num_features=num_features,
                          num_classes=num_classes,
                          fsdp_state_dict_type='full')

    trainer.fit()
    rankn_checkpoint = save_folder / pathlib.Path(f'rank{dist.get_global_rank()}.pt')

    # Check that rank 0 saves a checkpoint to disk, but rank 1 does not.
    if dist.get_global_rank() == 0:
        assert os.path.exists(rankn_checkpoint)
    elif dist.get_global_rank() == 1:
        assert not os.path.exists(rankn_checkpoint)
    state_dict_in_memory = trainer.state.state_dict()

    if dist.get_global_rank() == 0:
        # Check rank 0 state dict has the full model weights.
        assert set(state_dict_in_memory['model'].keys()) == {
            'module.2.weight', 'module.2.bias', 'module.4.weight', 'module.4.bias'
        }
        assert state_dict_in_memory['model']['module.2.weight'].ndim == 2
        assert state_dict_in_memory['model']['module.2.weight'].shape == layer1_weights_shape
        assert state_dict_in_memory['model']['module.2.bias'].shape == layer1_bias_shape
        assert state_dict_in_memory['model']['module.4.weight'].shape == layer2_weights_shape
        assert state_dict_in_memory['model']['module.4.bias'].shape == layer2_bias_shape
        assert sum([p.numel() for p in state_dict_in_memory['model'].values()]) == expected_total_num_params

        # Check rank 0 state dict also has the full optimizer params.
        optim_state_dict = state_dict_in_memory['optimizers']['Adam']['state']
        assert all([
            optim_moment.shape == layer1_weights_shape
            for moment_name, optim_moment in optim_state_dict['module.2.weight'].items()
            if moment_name != 'step'
        ])
        assert all([
            optim_moment.shape == layer2_weights_shape
            for moment_name, optim_moment in optim_state_dict['module.4.weight'].items()
            if moment_name != 'step'
        ])
        assert all([
            optim_moment.shape == layer1_bias_shape
            for moment_name, optim_moment in optim_state_dict['module.2.bias'].items()
            if moment_name != 'step'
        ])
        assert all([
            optim_moment.shape == layer2_bias_shape
            for moment_name, optim_moment in optim_state_dict['module.4.bias'].items()
            if moment_name != 'step'
        ])

        # Check that checkpoint matches state dict
        with open(str(rankn_checkpoint), 'rb') as f:
            state_dict_from_checkpoint = torch.load(f)['state']

        _compare_model_params_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)

        _compare_optims_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)

    if dist.get_global_rank() == 1:
        # Check rank 1 state dict just has the flattened shards.
        rank1_state_dict_keys = set(state_dict_in_memory['model'].keys())
        # Assert all params flattened
        assert all([k.endswith('flat_param') for k in rank1_state_dict_keys])
        assert all([p.ndim == 1 for p in state_dict_in_memory['model'].values()])
        # Assert total number of params is half of the total (because partitioned across 2 ranks).
        assert sum([p.numel() for p in state_dict_in_memory['model'].values()
                   ]) == expected_total_num_params / dist.get_world_size()

        # In FSDP for full state dicts, the optim state dicts on other ranks are empty dictionaries.
        assert state_dict_in_memory['optimizers']['Adam'] == {}


@pytest.mark.gpu
@world_size(2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load(world_size, tmp_path: pathlib.Path):
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer1 = get_trainer(save_folder=str(save_folder), save_filename=save_filename, fsdp_state_dict_type='full')
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(fsdp_state_dict_type='full', load_path=load_path)
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_partitioned_state_dict_save(world_size, tmp_path: pathlib.Path, state_dict_type: str):
    pytest.importorskip('torch.distributed.fsdp.fully_sharded_data_parallel')
    from torch.distributed.fsdp.fully_sharded_data_parallel import ShardedTensor
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'

    num_features = 3
    num_classes = 2

    expected_layer_shapes = [(5, num_features), (5,), (num_classes, 5), (num_classes,)]
    expected_total_num_params = sum([np.prod(shape) for shape in expected_layer_shapes])

    trainer = get_trainer(save_folder=str(save_folder),
                          save_filename=save_filename,
                          num_features=num_features,
                          num_classes=num_classes,
                          fsdp_state_dict_type=state_dict_type)

    trainer.fit()
    rankn_checkpoint = save_folder / pathlib.Path(f'rank{dist.get_global_rank()}.pt')

    # Check that both rank 0 and rank 1 save a checkpoint.
    assert os.path.exists(rankn_checkpoint)

    state_dict_in_memory = trainer.state.state_dict()

    if state_dict_type == 'local':
        rankn_state_dict_keys = set(state_dict_in_memory['model'].keys())
        # Assert all params flattened
        assert all([k.endswith('flat_param') for k in rankn_state_dict_keys])
        assert all([p.ndim == 1 for p in state_dict_in_memory['model'].values()])

        # Assert all params of type ShardedTensor.
        assert all([isinstance(p, ShardedTensor) for p in state_dict_in_memory['model'].values()])

        # Assert total number of params is half of the total (because partitioned across 2 ranks). Seems to divide evenly with flattened and sharded.
        assert sum([p.local_tensor().numel() for p in state_dict_in_memory['model'].values()
                   ]) == expected_total_num_params / dist.get_world_size()

        # Check optimizer is partitioned and flattened.
        rank_n_optim_state_dict = state_dict_in_memory['optimizers']['Adam']['state']
        # Assert all optim moments are flattened
        assert all([
            optim_moment.ndim == 1
            for module_name in rank_n_optim_state_dict.keys()
            for moment_name, optim_moment in rank_n_optim_state_dict[module_name].items()
            if moment_name != 'step'
        ])

        # Assert total number of moments in optim state divided across ranks.
        moments_per_parameter = 2
        assert sum([
            optim_moment.numel()
            for module_name in rank_n_optim_state_dict.keys()
            for moment_name, optim_moment in rank_n_optim_state_dict[module_name].items()
            if moment_name != 'step'
        ]) == (moments_per_parameter * expected_total_num_params) / dist.get_world_size()

    if state_dict_type == 'sharded':
        rankn_state_dict_keys = set(state_dict_in_memory['model'].keys())

        # Assert all params not flattened.
        assert not all([p.ndim == 1 for p in state_dict_in_memory['model'].values()])

        # Assert all params of type ShardedTensor
        assert all([isinstance(p, ShardedTensor) for p in state_dict_in_memory['model'].values()])

        # Assert total number of params is less than that of the total (because partitioned across 2 ranks). Does not divide
        # evenly with sharded and unflattened, so we just check that the params per rank is less than the total.
        assert sum([p.local_tensor().numel() for p in state_dict_in_memory['model'].values()
                   ]) < expected_total_num_params

        # Check optimizer is partitioned, but unflattened.
        rank_n_optim_state_dict = state_dict_in_memory['optimizers']['Adam']['state']
        # Assert all optim moments are flattened
        assert not all([
            optim_moment.ndim == 1
            for module_name in rank_n_optim_state_dict.keys()
            for moment_name, optim_moment in rank_n_optim_state_dict[module_name].items()
            if moment_name != 'step'
        ])

        # Assert total number of optim params is less than that of the total (because partitioned across 2 ranks). Does not divide
        # evenly with sharded and unflattened, so we just check that the optim params per rank is less than the total.
        moments_per_parameter = 2
        assert sum([
            optim_moment.local_tensor().numel()
            for module_name in rank_n_optim_state_dict.keys()
            for moment_name, optim_moment in rank_n_optim_state_dict[module_name].items()
            if moment_name != 'step'
        ]) < (moments_per_parameter * expected_total_num_params)

    # Check state dicts same between the in memory state and the on disk checkpoint for both ranks.
    with open(str(rankn_checkpoint), 'rb') as f:
        state_dict_from_checkpoint = torch.load(f)['state']

    _compare_model_params_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)

    _compare_optims_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_partitioned_state_dict_load(world_size, tmp_path: pathlib.Path, state_dict_type: str):
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer1 = get_trainer(save_folder=str(save_folder),
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type)
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(fsdp_state_dict_type=state_dict_type, load_path=load_path)
    state_dict_from_trainer2 = trainer2.state.state_dict()

    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
