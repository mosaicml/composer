# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import textwrap

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.algorithms import EMA
from composer.models import ComposerClassifier
from composer.trainer import Trainer
from composer.utils import dist
from tests.common import RandomClassificationDataset
from tests.common.compare import deep_compare
from tests.common import RandomClassificationDataset
from tests.common.markers import world_size
from composer.utils.file_helpers import get_file


# This model is to be used explicitly for this unit test because it some old reference checkpoints
# were saved using it exactly as it is. Changing this model will break test_fsdp_load_old_checkpoint.
class SimpleMLP(ComposerClassifier):

    def __init__(self, num_features: int=32, num_classes: int = 8):
        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_classes, bias=False),
        )
        super().__init__(module=net, num_classes=num_classes)


def get_trainer(
    save_folder=None,
    save_filename='ba{batch}-rank{rank}.pt',
    save_overwrite=False,
    num_features=2,
    num_classes=2,
    fsdp_state_dict_type='full',
    load_path=None,
    autoresume=False,
    run_name=None,
    max_duration='2ba',
    precision='amp_fp16',
    sharding_strategy='FULL_SHARD',
    save_interval='2ba',
    algorithms=None,
):
    model = SimpleMLP(num_features=num_features, num_classes=num_classes)
    dataset = RandomClassificationDataset(shape=(num_features,), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=32)
    optim = torch.optim.Adam(params=model.parameters())
    trainer = Trainer(
        algorithms=algorithms,
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'min_params': 16,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': sharding_strategy,
        },
        save_folder=save_folder,
        max_duration=max_duration,
        save_interval=save_interval,
        save_filename=save_filename,
        save_overwrite=save_overwrite,
        precision=precision,
        load_path=load_path,
        progress_bar=False,
        log_to_console=False,
        autoresume=autoresume,
        run_name=run_name,
        save_latest_filename='latest-rank{rank}.pt',
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


def _compare_metrics_between_state_dicts(state_dict1, state_dict2):
    # Check that metric states are equal between in memory mode and checkpoint
    state_dict1_train_metrics = state_dict1['train_metrics']
    state_dict2_train_metrics = state_dict2['train_metrics']

    state_dict1_eval_metrics = state_dict1['eval_metrics']
    state_dict2_eval_metrics = state_dict2['eval_metrics']

    deep_compare(state_dict1_train_metrics, state_dict2_train_metrics)
    deep_compare(state_dict1_eval_metrics, state_dict2_eval_metrics)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('autoresume', [True, False])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load(world_size, tmp_path: pathlib.Path, autoresume: bool, precision: str):
    if autoresume:
        run_name = 'my-cool-autoresume-run'
    else:
        run_name = None
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        load_path=load_path,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        max_duration='4ba',
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    # Continue to fit to make sure we can continue training.
    trainer2.fit()


@pytest.mark.gpu
@pytest.mark.remote
@world_size(2)
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD', 'SHARD_GRAD_OP'])
@pytest.mark.parametrize('state_dict_type', ['full', 'sharded', 'local'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_load_old_checkpoint(world_size, tmp_path: pathlib.Path, precision: str, sharding_strategy: str, state_dict_type: str):
    if version.parse(torch.__version__) >= version.parse('2.0.0') and state_dict_type == 'local':
        pytest.xfail("Loading a torch 1.13 checkpoint with torch 2.0 for state_dict_type local is not backwards compatible")
    

    rank = 0 if state_dict_type == 'full' else '{rank}'
    load_path = f's3://mosaicml-internal-checkpoints-test/ckpt-compatibility-test/{sharding_strategy.lower()}_{state_dict_type}_{precision}/rank{rank}.pt'

    trainer2 = get_trainer(
        fsdp_state_dict_type=state_dict_type,
        num_features=32, # This parameter setting is very important. Don't change or the test will fail.
        num_classes=8, # This parameter setting is very important. Don't change or the test will fail.
        sharding_strategy=sharding_strategy,
        load_path=load_path,
        precision=precision,
        max_duration='4ba',
    )
    state_dict2 = trainer2.state.state_dict()

    if (dist.get_global_rank() == 0 and  state_dict_type == 'full') or state_dict_type in ['sharded', 'local']:
        filled_load_path = load_path.format(rank=dist.get_global_rank())
        destination = str(tmp_path / pathlib.Path(filled_load_path).name)
        get_file(filled_load_path, destination=destination)
        with open(destination, 'rb') as f:
            state_dict1 = torch.load(f)['state']
        _compare_model_params_between_state_dicts(state_dict1, state_dict2)

        _compare_optims_between_state_dicts(state_dict1, state_dict2)

    # Continue to fit to make sure we can continue training.
    trainer2.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load_with_ema(world_size, tmp_path: pathlib.Path, precision: str):
    save_folder = tmp_path
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        sharding_strategy='SHARD_GRAD_OP',
        algorithms=EMA(smoothing=0.9999, half_life=None, update_interval='1ba'),
        save_interval='1ba',
        max_duration='5ba',
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()

    load_path = str(save_folder / pathlib.Path('ba4-rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type='full',
        load_path=load_path,
        sharding_strategy='SHARD_GRAD_OP',
        algorithms=EMA(smoothing=0.9999, half_life=None, update_interval='1ba'),
        save_interval='1ba',
        save_overwrite=True,
    )
    trainer2.fit(duration='1ba')
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('autoresume', [True, False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_partitioned_state_dict_load(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool,
                                          precision: str):
    if autoresume:
        run_name = 'my-autoresume-run'
    else:
        run_name = None
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'
    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type=state_dict_type,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        fsdp_state_dict_type=state_dict_type,
        load_path=load_path,
        precision=precision,
        autoresume=autoresume,
        run_name=run_name,
        max_duration='4ba',
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()

    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
    _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
    _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    trainer2.fit()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('autoresume', [True])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_mismatch_timestamp_error(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool):
    run_name = 'my-run-ar' if autoresume else 'my-run'
    save_folder = str(tmp_path / pathlib.Path(run_name))
    save_filename = 'ba{batch}-rank{rank}.pt'
    trainer1 = get_trainer(save_folder=save_folder,
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           run_name=run_name,
                           autoresume=autoresume,
                           max_duration='2ba',
                           save_interval='1ba')
    trainer1.fit()
    trainer1.close()
    # Corrupt latest checkpoint symlink for rank1 by changing it from batch 2 checkpoint to the batch 1 one
    # and removing batch 2 checkpoint.
    if dist.get_global_rank() == 1:
        latest_symlink = str(pathlib.Path(save_folder) / pathlib.Path('latest-rank1.pt'))
        latest_checkpoint_path = pathlib.Path(save_folder) / pathlib.Path(save_filename.format(batch=2, rank=1))
        assert os.readlink(latest_symlink) == latest_checkpoint_path.name
        oldest_checkpoint_path = pathlib.Path(save_folder) / pathlib.Path(save_filename.format(batch=1, rank=1))
        os.remove(latest_symlink)
        os.symlink(src=oldest_checkpoint_path.name, dst=latest_symlink)
        os.remove(latest_checkpoint_path)
        assert os.readlink(latest_symlink) == oldest_checkpoint_path.name

    expected_error = pytest.raises(RuntimeError, match='Timestamp mismatch error:*')

    with expected_error:
        get_trainer(
            save_folder=save_folder,
            save_filename=save_filename,
            fsdp_state_dict_type=state_dict_type,
            autoresume=autoresume,
            run_name=run_name,
        )
