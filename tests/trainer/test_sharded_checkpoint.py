# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import textwrap

import pytest
import torch
from torch.utils.data import DataLoader

from composer.trainer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.markers import world_size


def get_trainer(save_folder=None,
                save_filename='ba{batch}-rank{rank}.pt',
                num_features=2,
                fsdp_state_dict_type='full',
                load_path=None):
    model = SimpleModel(num_features=num_features)
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
        but instead these keys that belong to one, but not the other:
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
def test_fsdp_full_state_dict_save(world_size, tmp_path: pathlib.Path):

    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer = get_trainer(save_folder=str(save_folder), save_filename=save_filename, fsdp_state_dict_type='full')

    trainer.fit()
    rankn_checkpoint = save_folder / pathlib.Path(f'rank{dist.get_global_rank()}.pt')

    # Check that rank 0 saves a checkpoint to disk, but rank 1 does not.
    if dist.get_global_rank() == 0:
        assert os.path.exists(rankn_checkpoint)
    elif dist.get_global_rank() == 1:
        assert not os.path.exists(rankn_checkpoint)
    state_dict_in_memory = trainer.state.state_dict()
    if dist.get_global_rank() == 0:

        with open(str(rankn_checkpoint), 'rb') as f:
            state_dict_from_checkpoint = torch.load(f)['state']

        _compare_model_params_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)

        _compare_optims_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)


@pytest.mark.gpu
@world_size(2)
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
def test_fsdp_partitioned_state_dict_save(world_size, tmp_path: pathlib.Path, state_dict_type: str):

    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer = get_trainer(save_folder=str(save_folder),
                          save_filename=save_filename,
                          fsdp_state_dict_type=state_dict_type)

    trainer.fit()
    rankn_checkpoint = save_folder / pathlib.Path(f'rank{dist.get_global_rank()}.pt')

    # Check that both rank 0 and rank 1 save a checkpoint
    assert os.path.exists(rankn_checkpoint)

    state_dict_in_memory = trainer.state.state_dict()

    # Check state dicts same between the in memory state and the on disk checkpoint for both ranks.
    with open(str(rankn_checkpoint), 'rb') as f:
        state_dict_from_checkpoint = torch.load(f)['state']

    _compare_model_params_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)

    _compare_optims_between_state_dicts(state_dict_from_checkpoint, state_dict_in_memory)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
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
