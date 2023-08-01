# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import pathlib
import textwrap
import uuid
from functools import partial

import numpy as np
import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.algorithms import EMA
from composer.core.state import fsdp_get_optim_state_dict, fsdp_state_dict_type_context
from composer.models import ComposerClassifier
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from composer.utils import dist
from composer.utils.checkpoint import is_checkpoint_legacy_sharded
from composer.utils.file_helpers import get_file
from composer.utils.misc import using_torch_2
from composer.utils.object_store import S3ObjectStore
from composer.utils.reproducibility import get_rng_state
from tests.common import RandomClassificationDataset, deep_compare
from tests.common.compare import deep_compare
from tests.common.markers import world_size


# This model is to be used explicitly for this unit test because some old reference checkpoints
# were saved using it exactly as it is. Changing this model will break test_fsdp_load_old_checkpoint.
class SimpleMLP(ComposerClassifier):

    def __init__(self, num_features: int = 32, num_classes: int = 8):
        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_classes, bias=False),
        )
        net.param_init_fn = self.param_init_fn
        super().__init__(module=net, num_classes=num_classes)

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def get_trainer(
    model_init_device='cpu',
    save_folder=None,
    save_filename='ba{batch}-rank{rank}.pt',
    save_overwrite=False,
    num_features=2,
    num_classes=2,
    fsdp_state_dict_type='full',
    fsdp_sharded_ckpt_prefix_dir='ba{batch}',
    load_path=None,
    autoresume=False,
    run_name=None,
    max_duration='2ba',
    precision='amp_fp16',
    sharding_strategy='FULL_SHARD',
    save_interval='2ba',
    save_weights_only=False,
    load_weights_only=False,
    algorithms=None,
    optimizer='adam',
    load_fsdp_monolith_rank0_only=False,
    save_num_checkpoints_to_keep=-1,
    sync_module_states=True,
):
    model = SimpleMLP(num_features=num_features, num_classes=num_classes)
    model.to(model_init_device)
    dataset = RandomClassificationDataset(shape=(num_features,), size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=8)
    if optimizer == 'adam':
        optim = torch.optim.Adam(params=model.parameters())
    elif optimizer == 'adamw':
        optim = DecoupledAdamW(model.parameters())
    else:
        raise ValueError(f'Unsupported optimizer name {optimizer}')
    trainer = Trainer(
        algorithms=algorithms,
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        fsdp_config={
            'min_params': 16,
            'state_dict_type': fsdp_state_dict_type,
            'sharding_strategy': sharding_strategy,
            'sharded_ckpt_prefix_dir': fsdp_sharded_ckpt_prefix_dir,
            'sync_module_states': sync_module_states,
            'use_orig_params': False,
            'load_fsdp_monolith_rank0_only': load_fsdp_monolith_rank0_only,
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
        save_weights_only=save_weights_only,
        load_weights_only=load_weights_only,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
    )
    return trainer


def _compare_optims_between_state_dicts(state_dict1, state_dict2):
    # Check that optim params are equal between checkpoint and in memory optimizer
    assert len(list(state_dict1['optimizers'].keys())) == 1
    assert len(list(state_dict2['optimizers'].keys())) == 1
    optim_key1 = list(state_dict1['optimizers'].keys()).pop()
    optim_key2 = list(state_dict2['optimizers'].keys()).pop()
    assert optim_key1 == optim_key2
    state_dict1_optim_params = state_dict1['optimizers'][optim_key1]['state']
    state_dict2_optim_params = state_dict2['optimizers'][optim_key2]['state']
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
            state_dict1_moment = state_dict1_param_moment_dict[moment_name].cpu()
            state_dict2_moment = state_dict2_param_moment_dict[moment_name].cpu()
            assert torch.equal(
                state_dict1_moment, state_dict2_moment
            ), f'Moment {moment_name} for parameter {param_name} not the same between state dicts,\n\t{state_dict1_moment}\n\t{state_dict2_moment}'


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
        state_dict1_model_tensor = state_dict1_model_params[param_name].cpu()
        state_dict2_model_tensor = state_dict2_model_params[param_name].cpu()
        assert torch.equal(state_dict1_model_tensor,
                           state_dict2_model_tensor), f'Weight named {param_name} not the same between state_dicts'


def _compare_rng_states_between_trainers(rng_state1, rng_state2):
    assert len(rng_state1) == len(rng_state2)
    for rank, rank_state1, rank_state2 in zip(range(len(rng_state1)), rng_state1, rng_state2):
        rank_state1_keys = set(rank_state1.keys())
        rank_state2_keys = set(rank_state2.keys())
        assert len(rank_state1_keys.symmetric_difference(rank_state2_keys)) == 0, textwrap.dedent(
            f"""The two rank rng state dicts being compared for rank {rank} must have the exact same set of keys,
            but instead these keys that belong to one, but not the other:
            {rank_state1_keys.symmetric_difference(rank_state2_keys)}""")
        python_state1 = rank_state1['python']
        python_state2 = rank_state2['python']
        assert python_state1 == python_state2, f'Python rng state not the same between state_dicts for rank {rank}'

        numpy_state1 = rank_state1['numpy']
        numpy_state2 = rank_state2['numpy']
        _, keys1, pos1, has_gauss1, cached_gaussian1 = numpy_state1
        _, keys2, pos2, has_gauss2, cached_gaussian2 = numpy_state2
        assert np.allclose(keys1, keys2,
                           equal_nan=True), f'Numpy rng keys state not the same between state_dicts for rank {rank}'
        assert pos1 == pos2, f'Numpy rng pos state not the same between state_dicts for rank {rank}'
        assert has_gauss1 == has_gauss2, f'Numpy rng has_gauss state not the same between state_dicts for rank {rank}'
        assert cached_gaussian1 == cached_gaussian2, f'Numpy rng cached_gaussian state not the same between state_dicts for rank {rank}'

        torch_state1 = rank_state1['torch']
        torch_state2 = rank_state2['torch']
        assert torch.equal(torch_state1,
                           torch_state2), f'Torch rng state not the same between state_dicts for rank {rank}'

        if 'cuda' in rank_state1_keys:
            cuda_state1 = rank_state1['cuda']
            cuda_state2 = rank_state2['cuda']
            torch.equal(cuda_state1, cuda_state2), f'Cuda rng state not the same between state_dicts for rank {rank}'


def _compare_metrics_between_state_dicts(state_dict1, state_dict2):
    # Check that metric states are equal between in memory mode and checkpoint
    state_dict1_train_metrics = state_dict1['train_metrics']
    state_dict2_train_metrics = state_dict2['train_metrics']

    state_dict1_eval_metrics = state_dict1['eval_metrics']
    state_dict2_eval_metrics = state_dict2['eval_metrics']
    for metric1, metric2 in zip(state_dict1_train_metrics.values(), state_dict2_train_metrics.values()):
        assert metric1['_computed'] == metric2['_computed']

    for metric1, metric2 in zip(state_dict1_eval_metrics.values(), state_dict2_eval_metrics.values()):
        assert metric1['_computed'] == metric2['_computed']


def _compare_timestamps_between_state_dicts(state_dict1, state_dict2):
    timestamp1 = state_dict1['timestamp']
    timestamp2 = state_dict2['timestamp']
    deep_compare(timestamp1, timestamp2)


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
@pytest.mark.parametrize('autoresume', [True, False])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('load_fsdp_monolith_rank0_only', [True, False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load(world_size, tmp_path: pathlib.Path, autoresume: bool, precision: str, optimizer: str,
                                   load_fsdp_monolith_rank0_only: bool):

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
        optimizer=optimizer,
        load_fsdp_monolith_rank0_only=load_fsdp_monolith_rank0_only,
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
        optimizer=optimizer,
        load_fsdp_monolith_rank0_only=load_fsdp_monolith_rank0_only,
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
    # Continue to fit to make sure we can continue training.
    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('sync_module_states', [True, False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_mixed_with_sync(world_size, tmp_path: pathlib.Path, sync_module_states: bool):
    context = contextlib.nullcontext() if sync_module_states else pytest.raises(ValueError,
                                                                                match='Detected mixed initialization.*')
    with context:
        get_trainer(
            model_init_device=['cpu', 'meta'][dist.get_global_rank()],
            save_folder=str(tmp_path),
            fsdp_state_dict_type='full',
            sync_module_states=sync_module_states,
        )


@pytest.mark.gpu
@pytest.mark.remote
@world_size(2)
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD', 'SHARD_GRAD_OP'])
@pytest.mark.parametrize('state_dict_type', ['full', 'sharded', 'local'])
@pytest.mark.parametrize('composer_version', [
    pytest.param(
        '0.13.5',
        marks=[
            pytest.mark.filterwarnings(
                r'ignore:ShardedGradScaler is not in the state_dict. Its state will not be restored.:UserWarning'),
            pytest.mark.filterwarnings(
                r'ignore:MosaicMLLogger is not in the state_dict. Its state will not be restored.:UserWarning'),
        ]),
    pytest.param('0.14.0',
                 marks=pytest.mark.filterwarnings(
                     r'ignore:MosaicMLLogger is not in the state_dict. Its state will not be restored.:UserWarning')),
    pytest.param('0.14.1',
                 marks=pytest.mark.filterwarnings(
                     r'ignore:MosaicMLLogger is not in the state_dict. Its state will not be restored.:UserWarning')),
])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_load_old_checkpoint(world_size, tmp_path: pathlib.Path, precision: str, sharding_strategy: str,
                                  state_dict_type: str, s3_bucket: str, s3_read_only_prefix: str,
                                  composer_version: str):
    if version.parse(torch.__version__) >= version.parse('2.0.0') and state_dict_type == 'local':
        pytest.xfail(
            'Loading a torch 1.13 checkpoint with torch 2.0 for state_dict_type local is not backwards compatible. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )

    rank = 0 if state_dict_type == 'full' else '{rank}'
    load_path = f's3://{s3_bucket}/{s3_read_only_prefix}/backwards_compatibility/{composer_version}/{sharding_strategy.lower()}_{state_dict_type}_{precision}/ba2_rank{rank}.pt'

    assert is_checkpoint_legacy_sharded(object_store=S3ObjectStore(bucket=f'{s3_bucket}'),
                                        source_path=load_path.lstrip(f's3://{s3_bucket}/'))
    trainer = get_trainer(
        fsdp_state_dict_type=state_dict_type,
        num_features=32,  # This parameter setting is very important. Don't change or the test will fail.
        num_classes=8,  # This parameter setting is very important. Don't change or the test will fail.
        sharding_strategy=sharding_strategy,
        load_path=load_path,
        precision=precision,
        max_duration='4ba',
    )
    state_dict2 = trainer.state.state_dict()

    if (dist.get_global_rank() == 0 and state_dict_type == 'full') or state_dict_type in ['sharded', 'local']:
        filled_load_path = load_path.format(rank=dist.get_global_rank())
        destination = str(tmp_path / pathlib.Path(filled_load_path).name)
        get_file(filled_load_path, destination=destination)
        with open(destination, 'rb') as f:
            state_dict1 = torch.load(f)['state']
        _compare_model_params_between_state_dicts(state_dict1, state_dict2)

        _compare_optims_between_state_dicts(state_dict1, state_dict2)

    # Continue to fit to make sure we can continue training.
    trainer.fit()
    trainer.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_fsdp_full_state_dict_load_with_ema(world_size, tmp_path: pathlib.Path, precision: str, optimizer: str):
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
        optimizer=optimizer,
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
        optimizer=optimizer,
    )
    trainer2.fit(duration='1ba')
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('weights_only', [False, True])
@pytest.mark.parametrize('optimizer', ['adam', 'adamw'])
@pytest.mark.parametrize('state_dict_type', ['sharded', 'local'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('use_remote', [pytest.param(True, marks=pytest.mark.remote), False])
@pytest.mark.parametrize('autoresume', [True, False])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
def test_fsdp_partitioned_state_dict_load(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool,
                                          precision: str, optimizer: str, weights_only: bool, use_remote, s3_bucket,
                                          s3_ephemeral_prefix, request):
    if weights_only and autoresume:
        pytest.xfail('Weights only with autoresume is not supported')
    if state_dict_type == 'local' and using_torch_2():
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )

    if autoresume:
        local_run_name = f'my-cool-autoresume-run-{uuid.uuid1()}'
        run_name = dist.all_gather_object(local_run_name)[0]
    else:
        run_name = None

    if use_remote:
        save_folder = f's3://{s3_bucket}/{s3_ephemeral_prefix}/checkpoints/{{run_name}}'
    else:
        tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
        save_folder = os.path.join(tmp_paths[0], 'checkpoints', '{run_name}')

    save_filename = 'ba{batch}-rank{rank}.pt'

    trainer1 = get_trainer(save_folder=str(save_folder),
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           run_name=run_name,
                           precision=precision,
                           autoresume=autoresume,
                           optimizer=optimizer,
                           max_duration='2ba',
                           save_interval='2ba',
                           save_weights_only=weights_only,
                           fsdp_sharded_ckpt_prefix_dir='ba{batch}')
    run_name = trainer1.state.run_name
    print(run_name)
    trainer1.fit()
    rng1 = get_rng_state()
    state_dict_from_trainer1_ba2 = trainer1.state.state_dict()
    trainer1.close()

    if use_remote:
        load_path = 's3://' + save_folder.strip('s3://').format(run_name=run_name) + '/ba2'
        object_store = S3ObjectStore(bucket=f'{s3_bucket}')
    else:
        object_store = None
        load_path = str(save_folder.format(run_name=run_name) / pathlib.Path('ba2'))

    if not using_torch_2():
        load_filename = f"{save_filename.format(batch=2, rank='{rank}')}"
        assert load_filename == 'ba2-rank{rank}.pt'
        load_path += '/' + load_filename
        assert is_checkpoint_legacy_sharded(object_store=object_store,
                                            source_path=load_path.replace(f's3://{s3_bucket}/', ''))
    else:
        assert not is_checkpoint_legacy_sharded(object_store=object_store,
                                                source_path=load_path.replace(f's3://{s3_bucket}/', ''))

    if autoresume:
        load_path = None
    trainer2 = get_trainer(save_folder=str(save_folder),
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           load_path=load_path,
                           precision=precision,
                           autoresume=autoresume,
                           run_name=run_name,
                           max_duration='4ba',
                           save_interval='4ba',
                           optimizer=optimizer,
                           load_weights_only=weights_only)
    state_dict_from_trainer2 = trainer2.state.state_dict()
    rng2 = trainer2._rng_state
    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(state_dict_from_trainer1_ba2, state_dict_from_trainer2)
    if not weights_only:
        _compare_rng_states_between_trainers(rng1, rng2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1_ba2, state_dict_from_trainer2)
        _compare_metrics_between_state_dicts(state_dict_from_trainer1_ba2, state_dict_from_trainer2)
        _compare_timestamps_between_state_dicts(state_dict_from_trainer1_ba2, state_dict_from_trainer2)

    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@pytest.mark.remote
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['sharded'])
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('autoresume', [False, True])  # True commented out for now
@pytest.mark.parametrize('num_shards', [2, 4, 7])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD', 'SHARD_GRAD_OP'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.0.1'), reason='requires PyTorch 2.01 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:MosaicMLLogger is not in the state_dict.:UserWarning')
def test_elastic_resumption(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool, precision: str,
                            sharding_strategy, s3_bucket, s3_read_only_prefix, num_shards: int):
    if state_dict_type == 'local' and using_torch_2():
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )
    if autoresume:
        run_name = 'my-autoresume-run'
    else:
        run_name = None

    base_path = f's3://{s3_bucket}/{s3_read_only_prefix}/elastic_test/{sharding_strategy.lower()}_{state_dict_type}_{precision}_{num_shards}/'

    mono_load_path = os.path.join(base_path, 'mono.pt')
    mono_trainer = get_trainer(
        fsdp_state_dict_type='full',
        load_path=mono_load_path,
        num_features=32,  # This parameter setting is very important. Don't change or the test will fail.
        num_classes=8,  # This parameter setting is very important. Don't change or the test will fail.
        precision=precision,
        autoresume=
        False,  # Hardcoded to false b/c checkpoints saved with the mono checkpoint saver callback don't have symlinks to them.
        run_name=run_name,
        max_duration='4ba',
    )
    if autoresume:
        sharded_load_path = None
        save_folder = base_path
    else:
        save_folder = None
        sharded_load_path = os.path.join(base_path, 'ba2')
        assert not is_checkpoint_legacy_sharded(object_store=S3ObjectStore(bucket=f'{s3_bucket}'),
                                                source_path=sharded_load_path.replace(f's3://{s3_bucket}/', ''))

    sharded_trainer = get_trainer(
        fsdp_state_dict_type=state_dict_type,
        save_folder=save_folder,
        load_path=sharded_load_path,
        num_features=
        32,  # This parameter setting is very important. It is hardcoded to match the num_features in the checkpoint it is downloading.
        num_classes=
        8,  # This parameter setting is very important. It is hardcoded to match the num_classes in the checkpoint it is downloading.
        precision=precision,
        autoresume=autoresume,
        run_name=run_name,
        max_duration='4ba',
        load_weights_only=False)

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
                fsdp_get_optim_state_dict(trainer.state.model, optimizer, state_dict_type='full')
        }
        return state_dict

    def compare_state_dicts():
        state_dict_from_trainer1 = mono_trainer.state.state_dict()
        state_dict_from_trainer2 = get_mono_state_dict_from_sharded_one(sharded_trainer)
        _compare_model_params_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_optims_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
        _compare_timestamps_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)

    # Compare state dicts.
    compare_state_dicts()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['local', 'sharded'])
@pytest.mark.parametrize('autoresume', [True])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
def test_mismatch_timestamp_error(world_size, tmp_path: pathlib.Path, state_dict_type: str, autoresume: bool):
    if state_dict_type == 'local' and using_torch_2():
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )
    run_name = 'my-run-ar' if autoresume else 'my-run'
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = str(tmp_paths[0] / pathlib.Path(run_name))
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
    latest_symlink = str(pathlib.Path(save_folder) / pathlib.Path(f'latest-rank{dist.get_global_rank()}.pt'))
    latest_checkpoint_path = pathlib.Path(save_folder) / pathlib.Path('ba2') / (pathlib.Path(
        save_filename.format(batch=2, rank=dist.get_global_rank())) if not using_torch_2() else pathlib.Path(''))
    assert os.path.join(save_folder, os.readlink(latest_symlink)) == str(latest_checkpoint_path)
    oldest_checkpoint_relative_path = str(
        pathlib.Path('ba1') / (pathlib.Path(save_filename.format(batch=1, rank=dist.get_global_rank()))
                               if not using_torch_2() else pathlib.Path('')))

    # Corrupt latest checkpoint symlink for rank1 by changing it from batch 2 checkpoint to the batch 1 one
    # and removing batch 2 checkpoint.
    if dist.get_global_rank() == 0:
        os.remove(latest_symlink)
        os.symlink(src=oldest_checkpoint_relative_path, dst=latest_symlink)
        assert os.readlink(latest_symlink) == oldest_checkpoint_relative_path

    dist.barrier()
    expected_error = pytest.raises(RuntimeError, match='Timestamp mismatch error:*')

    with expected_error:
        get_trainer(
            save_folder=save_folder,
            save_filename=save_filename,
            fsdp_state_dict_type=state_dict_type,
            autoresume=autoresume,
            run_name=run_name,
        )


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('state_dict_type', ['sharded', 'local'])
@pytest.mark.parametrize('num_ckpts_to_keep', [-1, 1, 2, 3])
@pytest.mark.parametrize('batches_to_train', [3])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
def test_cleanup_sharded_checkpoints(world_size, tmp_path: pathlib.Path, state_dict_type: str, num_ckpts_to_keep: int,
                                     batches_to_train: int, s3_bucket, s3_ephemeral_prefix, request):
    if state_dict_type == 'local' and using_torch_2():
        pytest.xfail(
            'Loading a state_dict_type="local" checkpoint with strict=True errors out. See https://github.com/pytorch/pytorch/issues/102667 for more info'
        )

    run_name = None

    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = os.path.join(tmp_paths[0], 'checkpoints', '{run_name}')
    save_filename = 'rank{rank}.pt'
    fsdp_sharded_ckpt_prefix_dir = 'ba{batch}'

    trainer1 = get_trainer(save_folder=str(save_folder),
                           save_filename=save_filename,
                           fsdp_state_dict_type=state_dict_type,
                           run_name=run_name,
                           max_duration=f'{batches_to_train}ba',
                           save_interval='1ba',
                           save_num_checkpoints_to_keep=num_ckpts_to_keep,
                           fsdp_sharded_ckpt_prefix_dir=fsdp_sharded_ckpt_prefix_dir)
    run_name = trainer1.state.run_name
    trainer1.fit()
    trainer1.close()

    shards_dir = os.path.join(save_folder.format(run_name=run_name))
    dir_contents = [file_or_dir for file_or_dir in os.listdir(shards_dir) if 'latest' not in file_or_dir]
    num_checkpoint_dirs = len(dir_contents)
    if num_ckpts_to_keep == -1:
        assert num_checkpoint_dirs == batches_to_train
    else:
        assert num_checkpoint_dirs == num_ckpts_to_keep

    for ckpt_dir in dir_contents:
        full_path_ckpt_dir = os.path.join(shards_dir, ckpt_dir)
        elastic_file_list = {'.metadata', *[f'__{rank}_0.distcp' for rank in range(dist.get_world_size())]}
        non_elastic_file_list = {save_filename.format(rank=rank) for rank in range(dist.get_world_size())}
        file_list = elastic_file_list if using_torch_2() else non_elastic_file_list
        assert set(os.listdir(full_path_ckpt_dir)) == file_list
