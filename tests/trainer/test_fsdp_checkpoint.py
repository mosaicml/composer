# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import dataclasses
import os
import pathlib
import textwrap
import uuid
from contextlib import nullcontext as does_not_raise
from functools import partial
from typing import Any, Callable, Optional, Sequence, Union
from unittest.mock import patch

import numpy as np
import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from composer.algorithms import EMA
from composer.core import Algorithm, Event, Precision, State, Time
from composer.core.state import fsdp_get_optim_state_dict, fsdp_state_dict_type_context
from composer.models import ComposerClassifier
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from composer.utils import dist, parse_uri
from composer.utils.checkpoint import is_checkpoint_legacy_sharded
from composer.utils.file_helpers import get_file
from composer.utils.object_store import S3ObjectStore
from composer.utils.reproducibility import get_rng_state
from tests.common import RandomClassificationDataset, deep_compare
from tests.common.compare import deep_compare
from tests.common.markers import world_size


# This model is to be used explicitly for this unit test because some old reference checkpoints
# were saved using it exactly as it is. Changing this model will break test_fsdp_load_old_checkpoint.
class SimpleMLP(ComposerClassifier):

    def __init__(
        self,
        num_features: int = 32,
        num_classes: int = 8,
        train_metrics: Optional[Metric | MetricCollection] = None,
        val_metrics: Optional[Metric | MetricCollection] = None,
    ):
        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_classes, bias=False),
        )

        for module in net:
            if isinstance(module, torch.nn.Linear):
                module._fsdp_wrap = True  # pyright: ignore[reportGeneralTypeIssues]

        net.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]
        super().__init__(
            module=net,
            num_classes=num_classes,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


@dataclasses.dataclass(frozen=True)
class FSDPConfig:
    state_dict_type: str = 'full'
    sharding_strategy: str = 'FULL_SHARD'
    sharded_ckpt_prefix_dir: str = 'ba{batch}'
    sync_module_states: bool = True
    use_orig_params: bool = False
    load_fsdp_monolith_rank0_only: bool = False
    save_planner: Optional[Any] = None
    load_planner: Optional[Any] = None


def get_trainer(
    model_init_device: str = 'cpu',
    save_folder: Optional[str] = None,
    save_filename: str = 'ba{batch}-rank{rank}.pt',
    save_overwrite: bool = False,
    num_features: int = 2,
    num_classes: int = 2,
    load_path: Optional[str] = None,
    autoresume: bool = False,
    run_name: Optional[str] = None,
    max_duration: Optional[int | str | Time] = '2ba',
    precision: Optional[str | Precision] = 'amp_fp16',
    save_interval: str | int | Time | Callable[[State, Event], bool] = '2ba',
    save_weights_only: bool = False,
    load_weights_only: bool = False,
    load_ignore_keys: Optional[list[str] | Callable[[dict], None]] = None,
    algorithms: Optional[Algorithm | Sequence[Algorithm]] = None,
    optimizer: str = 'adam',
    save_num_checkpoints_to_keep: int = -1,
    train_metrics: Optional[Any] = None,
    val_metrics: Optional[Any] = None,
    fsdp_config: Optional[FSDPConfig] = None,
):
    if fsdp_config is None:
        fsdp_config = FSDPConfig()
    model = SimpleMLP(
        num_features=num_features,
        num_classes=num_classes,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
    )
    model.to(model_init_device)
    dataset = RandomClassificationDataset(shape=(num_features,), size=128)
    dataloader = DataLoader(
        dataset,
        sampler=dist.get_sampler(dataset),
        batch_size=8,
    )
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
        fsdp_config=dataclasses.asdict(fsdp_config),
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
        load_ignore_keys=load_ignore_keys,
    )
    return trainer


def _compare_optims_between_state_dicts(state_dict1, state_dict2):
    # Check that optim params are equal between checkpoint and in memory optimizer
    assert len(list(state_dict1['optimizers'].keys())) == 1
    assert len(list(state_dict2['optimizers'].keys())) == 1
    optim_key1 = list(state_dict1['optimizers'].keys()).pop()
    optim_key2 = list(state_dict2['optimizers'].keys()).pop()
    assert optim_key1 == optim_key2
    # Note for state_dict_type = 'full' on nonzero ranks
    # state_dict1['optimizers'][optim_key1] and state_dict2['optimizers'][optim_key2] may be empty dictionaries.
    state_dict1_optim_params = state_dict1['optimizers'][optim_key1]['state']
    state_dict2_optim_params = state_dict2['optimizers'][optim_key2]['state']
    state_dict1_keys = set(state_dict1_optim_params.keys())
    state_dict2_keys = set(state_dict2_optim_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""",
    )

    for param_name in state_dict2_optim_params.keys():
        state_dict1_param_moment_dict = state_dict1_optim_params[param_name]
        state_dict2_param_moment_dict = state_dict2_optim_params[param_name]
        for moment_name in state_dict2_param_moment_dict.keys():
            state_dict1_moment = state_dict1_param_moment_dict[moment_name].cpu()
            state_dict2_moment = state_dict2_param_moment_dict[moment_name].cpu()
            assert torch.equal(state_dict1_moment, state_dict2_moment), (
                f'Moment {moment_name} for parameter {param_name} not the same '
                'between state dicts,\n\t{state_dict1_moment}\n\t'
                '{state_dict2_moment}'
            )


def _compare_model_params_between_state_dicts(state_dict1, state_dict2):
    # Check that model params are equal between in memory mode and checkpoint
    state_dict1_model_params = state_dict1['model']
    state_dict2_model_params = state_dict2['model']

    state_dict1_keys = set(state_dict1_model_params.keys())
    state_dict2_keys = set(state_dict2_model_params.keys())
    assert len(state_dict1_keys.symmetric_difference(state_dict2_keys)) == 0, textwrap.dedent(
        f"""The two state dicts being compared must have the exact same set of keys,
        but instead these keys that belong to one, but not the other:
        {state_dict1_keys.symmetric_difference(state_dict2_keys)}""",
    )

    for param_name in state_dict2_model_params.keys():
        state_dict1_model_tensor = state_dict1_model_params[param_name].cpu()
        state_dict2_model_tensor = state_dict2_model_params[param_name].cpu()
        assert torch.equal(
            state_dict1_model_tensor,
            state_dict2_model_tensor,
        ), f'Weight named {param_name} not the same between state_dicts'


def _compare_rng_states_between_trainers(rng_state1, rng_state2):
    assert len(rng_state1) == len(rng_state2)
    for rank, rank_state1, rank_state2 in zip(range(len(rng_state1)), rng_state1, rng_state2):
        rank_state1_keys = set(rank_state1.keys())
        rank_state2_keys = set(rank_state2.keys())
        assert len(rank_state1_keys.symmetric_difference(rank_state2_keys)) == 0, textwrap.dedent(
            f"""The two rank rng state dicts being compared for rank {rank} must have the exact same set of keys,
            but instead these keys that belong to one, but not the other:
            {rank_state1_keys.symmetric_difference(rank_state2_keys)}""",
        )
        python_state1 = rank_state1['python']
        python_state2 = rank_state2['python']
        assert python_state1 == python_state2, f'Python rng state not the same between state_dicts for rank {rank}'

        numpy_state1 = rank_state1['numpy']
        numpy_state2 = rank_state2['numpy']
        _, keys1, pos1, has_gauss1, cached_gaussian1 = numpy_state1
        _, keys2, pos2, has_gauss2, cached_gaussian2 = numpy_state2
        assert np.allclose(
            keys1,
            keys2,
            equal_nan=True,
        ), f'Numpy rng keys state not the same between state_dicts for rank {rank}'
        assert pos1 == pos2, f'Numpy rng pos state not the same between state_dicts for rank {rank}'
        assert has_gauss1 == has_gauss2, f'Numpy rng has_gauss state not the same between state_dicts for rank {rank}'
        assert cached_gaussian1 == cached_gaussian2, f'Numpy rng cached_gaussian state not the same between state_dicts for rank {rank}'

        torch_state1 = rank_state1['torch']
        torch_state2 = rank_state2['torch']
        assert torch.equal(
            torch_state1,
            torch_state2,
        ), f'Torch rng state not the same between state_dicts for rank {rank}'

        if 'cuda' in rank_state1_keys:
            cuda_state1 = rank_state1['cuda']
            cuda_state2 = rank_state2['cuda']
            states_equal = torch.equal(cuda_state1, cuda_state2)
            assert states_equal, f'Cuda rng state not the same between state_dicts for rank {rank}'


def _compare_metrics_between_state_dicts(state_dict1: dict[str, Any], state_dict2: dict[str, Any]):
    # Check that metric states are equal between in memory mode and checkpoint
    state_dict1_train_metrics = state_dict1.get('train_metrics', None)
    state_dict2_train_metrics = state_dict2.get('train_metrics', None)

    state_dict1_eval_metrics = state_dict1.get('eval_metrics', None)
    state_dict2_eval_metrics = state_dict2.get('eval_metrics', None)

    if state_dict1_train_metrics is not None and state_dict2_train_metrics is not None:
        for metric1, metric2 in zip(state_dict1_train_metrics.values(), state_dict2_train_metrics.values()):
            assert metric1['_computed'] == metric2['_computed']
    else:
        assert state_dict1_train_metrics == state_dict2_train_metrics

    if state_dict1_eval_metrics is not None and state_dict2_eval_metrics is not None:
        for metric1, metric2 in zip(state_dict1_eval_metrics.values(), state_dict2_eval_metrics.values()):
            assert metric1['_computed'] == metric2['_computed']
    else:
        assert state_dict1_eval_metrics == state_dict2_eval_metrics


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
def test_fsdp_full_state_dict_load(
    world_size,
    tmp_path: pathlib.Path,
    autoresume: bool,
    precision: str,
    optimizer: str,
    load_fsdp_monolith_rank0_only: bool,
):

    if autoresume:
        run_name = 'my-cool-autoresume-run'
    else:
        run_name = None
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'

    fsdp_config = FSDPConfig(load_fsdp_monolith_rank0_only=load_fsdp_monolith_rank0_only)

    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        optimizer=optimizer,
        fsdp_config=fsdp_config,
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()
    load_path = str(save_folder / pathlib.Path('rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        load_path=load_path,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        max_duration='4ba',
        optimizer=optimizer,
        fsdp_config=fsdp_config,
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(
            state_dict_from_trainer1,
            state_dict_from_trainer2,
        )
        _compare_optims_between_state_dicts(
            state_dict_from_trainer1,
            state_dict_from_trainer2,
        )
        _compare_metrics_between_state_dicts(
            state_dict_from_trainer1,
            state_dict_from_trainer2,
        )
    # Continue to fit to make sure we can continue training.
    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('sync_module_states', [True, False])
def test_fsdp_mixed_with_sync(
    world_size,
    tmp_path: pathlib.Path,
    sync_module_states: bool,
):
    context = contextlib.nullcontext()
    if not sync_module_states:
        context = pytest.raises(ValueError, match='Detected mixed initialization.*')

    with context:
        get_trainer(
            model_init_device=['cpu', 'meta'][dist.get_global_rank()],
            save_folder=str(tmp_path),
            fsdp_config=FSDPConfig(sync_module_states=sync_module_states),
        )


@pytest.mark.gpu
@pytest.mark.remote
@world_size(2)
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD', 'SHARD_GRAD_OP'])
@pytest.mark.parametrize('state_dict_type', ['full', 'sharded'])
@pytest.mark.parametrize(
    'composer_version',
    [
        pytest.param(
            '0.13.5',
            marks=[
                pytest.mark.filterwarnings(
                    (r'ignore:ShardedGradScaler is not in the state_dict. Its state will not be restored.:UserWarning'),
                ),
                pytest.mark.filterwarnings((
                    r'ignore:MosaicMLLogger is not in the state_dict. Its state '
                    r'will not be restored.:UserWarning'
                )),
            ],
        ),
        pytest.param(
            '0.14.0',
            marks=pytest.mark.filterwarnings(
                (r'ignore:MosaicMLLogger is not in the state_dict. Its '
                 r'state will not be restored.:UserWarning'),
            ),
        ),
        pytest.param(
            '0.14.1',
            marks=pytest.mark.filterwarnings(
                (r'ignore:MosaicMLLogger is not in the state_dict. Its '
                 r'state will not be restored.:UserWarning'),
            ),
        ),
        pytest.param(
            '0.15.1',
            marks=pytest.mark.filterwarnings(
                (r'ignore:MosaicMLLogger is not in the state_dict. Its '
                 r'state will not be restored.:UserWarning'),
            ),
        ),
        pytest.param(
            '0.16.0',
            marks=pytest.mark.filterwarnings(
                (r'ignore:MosaicMLLogger is not in the state_dict. Its '
                 r'state will not be restored.:UserWarning'),
            ),
        ),
        pytest.param(
            '0.17.0',
            marks=pytest.mark.filterwarnings(
                (r'ignore:MosaicMLLogger is not in the state_dict. Its '
                 r'state will not be restored.:UserWarning'),
            ),
        ),
        '0.18.1',
    ],
)
@pytest.mark.filterwarnings(r'ignore:.*metrics are not saved with sharded state dict.*:UserWarning')
@pytest.mark.filterwarnings(r'ignore:.*The CUDA RNG state could not be loaded.*:UserWarning')
def test_fsdp_load_old_checkpoint(
    world_size,
    tmp_path: pathlib.Path,
    precision: str,
    sharding_strategy: str,
    state_dict_type: str,
    s3_bucket: str,
    s3_read_only_prefix: str,
    composer_version: str,
):
    if composer_version == '0.18.1' and state_dict_type == 'full' and precision == 'amp_bf16' and sharding_strategy == 'FULL_SHARD':
        pytest.skip('TODO: This checkpoint is missing')

    if composer_version in ['0.13.5', '0.14.0', '0.14.1', '0.15.1']:
        rank = 0 if state_dict_type == 'full' else '{rank}'

        load_path_dir = (
            f's3://{s3_bucket}/{s3_read_only_prefix}/backwards_compatibility/'
            f'{composer_version}/{sharding_strategy.lower()}_{state_dict_type}_'
            f'{precision}/'
        )
        if ((version.parse(composer_version) > version.parse('0.15.0')) and state_dict_type != 'full'):
            load_path_dir = (load_path_dir + 'ep0-ba2/')

        load_path = load_path_dir + f'ba2_rank{rank}.pt'
        assert is_checkpoint_legacy_sharded(
            object_store=S3ObjectStore(bucket=f'{s3_bucket}'),
            source_path=load_path.lstrip(f's3://{s3_bucket}/'),
        )
    else:
        load_path = (
            f's3://{s3_bucket}/{s3_read_only_prefix}/backwards_compatibility/'
            f'{composer_version}/{sharding_strategy.lower()}_{state_dict_type}_'
            f'{precision}/'
        )
        if state_dict_type == 'full':
            load_path += 'ba2_rank0.pt'
        else:
            load_path += 'ep0-ba2/'

    if composer_version == '0.15.1':
        num_classes = 8  # This parameter setting is very important. Don't change or the test will fail.
        train_metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes),
        ])
        val_metrics = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes),
        ])
    else:
        train_metrics = None
        val_metrics = None

    fsdp_config = FSDPConfig(
        state_dict_type=state_dict_type,
        sharding_strategy=sharding_strategy,
    )

    trainer = get_trainer(
        num_features=32,  # This parameter setting is very important. Don't change or the test will fail.
        num_classes=8,  # This parameter setting is very important. Don't change or the test will fail.
        load_path=load_path,
        precision=precision,
        max_duration='4ba',
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        fsdp_config=fsdp_config,
    )
    state_dict2 = trainer.state.state_dict()

    if (dist.get_global_rank() == 0 and state_dict_type == 'full') or state_dict_type == 'sharded':
        # After composer version 0.16.0, sharded checkpoints are of type folder/__{local_rank}__{global_rank}.distcp
        # They cannot be loaded with `get_file` as we need the whole folder to load the checkpoint.
        # Thus, we use the DistCPObjectStoreReader to load the state_dict.
        if state_dict_type == 'sharded' and version.parse(composer_version) >= version.parse('0.16.0'):
            trainer2 = get_trainer(
                num_features=32,  # This parameter setting is very important. Don't change or the test will fail.
                num_classes=8,  # This parameter setting is very important. Don't change or the test will fail.
                precision=precision,
                max_duration='10ba',  # Change this so we have slightly different model runtime settings.
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                fsdp_config=fsdp_config,
            )

            from torch.distributed import checkpoint as dist_cp

            from composer.utils.checkpoint import DistCPObjectStoreReader

            _, _, parsed_load_path = parse_uri(load_path)
            gathered_tmp_path = str(dist.all_gather_object(tmp_path)[0])
            destination = str(pathlib.Path(gathered_tmp_path) / parsed_load_path)
            state_dict: dict[str, Any] = {
                'state': trainer2.state.state_dict(),
                'rng': get_rng_state(),
            }
            if version.parse(torch.__version__) < version.parse('2.2.9'):
                state_dict['state'].pop('optimizers')

            object_store = S3ObjectStore(bucket=f'{s3_bucket}')
            storage_reader = DistCPObjectStoreReader(
                source_path=parsed_load_path,
                destination_path=destination,
                object_store=object_store,
                device_mesh=None,
            )

            process_group = None
            dist_cp.load_state_dict(
                state_dict=state_dict,
                storage_reader=storage_reader,
                planner=None,
                process_group=process_group,
            )
            if version.parse(torch.__version__) < version.parse('2.2.9'):
                from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                model_state_dict = state_dict['state']['model']
                model = trainer2.state.model
                optim = trainer2.state.optimizers[0]
                optim_name = type(optim).__qualname__
                optim_state_dict = load_sharded_optimizer_state_dict(
                    model_state_dict=model_state_dict,
                    optimizer_key='optimizers',
                    storage_reader=storage_reader,
                )
                with fsdp_state_dict_type_context(module=model, state_dict_type=state_dict_type):
                    optim_state_dict = FSDP.optim_state_dict_to_load(
                        optim_state_dict=optim_state_dict['optimizers'][optim_name],
                        model=model,
                        optim=optim,
                    )

                trainer2.state.optimizers[0].load_state_dict(optim_state_dict)

                with fsdp_state_dict_type_context(module=model, state_dict_type=state_dict_type):
                    flattened_optim_state_dict = FSDP.optim_state_dict(model, optim)  # type: ignore

                state_dict['state']['optimizers'] = {
                    optim_name: flattened_optim_state_dict,
                }

            state_dict1 = state_dict['state']
        else:
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
def test_fsdp_full_state_dict_load_with_ema(
    world_size,
    tmp_path: pathlib.Path,
    precision: str,
    optimizer: str,
):
    save_folder = tmp_path
    save_filename = 'ba{batch}-rank{rank}.pt'

    fsdp_config = FSDPConfig(sharding_strategy='SHARD_GRAD_OP')

    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        algorithms=EMA(smoothing=0.9999, half_life=None, update_interval='1ba'),
        save_interval='1ba',
        max_duration='5ba',
        optimizer=optimizer,
        fsdp_config=fsdp_config,
    )
    trainer1.fit()
    state_dict_from_trainer1 = trainer1.state.state_dict()
    trainer1.close()

    load_path = str(save_folder / pathlib.Path('ba4-rank{rank}.pt'))
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        load_path=load_path,
        algorithms=EMA(smoothing=0.9999, half_life=None, update_interval='1ba'),
        save_interval='1ba',
        save_overwrite=True,
        optimizer=optimizer,
        fsdp_config=fsdp_config,
    )
    trainer2.fit(duration='1ba')
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(
            state_dict_from_trainer1,
            state_dict_from_trainer2,
        )
        _compare_optims_between_state_dicts(
            state_dict_from_trainer1,
            state_dict_from_trainer2,
        )

    trainer2.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('is_valid_checkpoint', [True, False])
@pytest.mark.parametrize('state_dict_type', ['sharded', 'full'])
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:.*metrics are not saved with sharded state dict.*:UserWarning')
@pytest.mark.filterwarnings(r'ignore:Please use DTensor instead and we are deprecating ShardedTensor.:UserWarning')
def test_checkpoint_loading_with_validation(world_size, tmp_path, is_valid_checkpoint: bool, state_dict_type: str):
    # Set the error expectations.
    expectation = does_not_raise()
    if not is_valid_checkpoint:
        expectation = pytest.raises(ValueError)

    def mock_get_checkpoint_validation_function():
        return lambda _: is_valid_checkpoint

    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = os.path.join(tmp_paths[0], 'checkpoints')
    fsdp_config = FSDPConfig(state_dict_type=state_dict_type)

    # First trainer saves checkpoints.
    trainer = get_trainer(save_folder=save_folder, fsdp_config=fsdp_config, max_duration='1ba')
    trainer.fit()
    trainer.close()

    # Determine the checkpoint path for loading.
    checkpoint_relpath = 'ba1-rank0.pt'
    if state_dict_type == 'sharded':
        checkpoint_relpath = 'ba1'

    # Load checkpoints with checkpoint validation.
    with expectation:
        with patch(
            'composer.utils.checkpoint._get_checkpoint_validation_function',
            mock_get_checkpoint_validation_function,
        ):
            trainer = get_trainer(
                load_path=os.path.join(save_folder, checkpoint_relpath),
                max_duration='2ba',
                fsdp_config=fsdp_config,
            )
            trainer.fit()
            trainer.close()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('use_remote', [pytest.param(True, marks=pytest.mark.remote), False])
@pytest.mark.parametrize(
    'weights_only,optimizer,precision,autoresume,load_ignore_keys,use_symlink',
    [
        [False, 'adamw', 'amp_bf16', False, None, True],
        [False, 'adamw', 'amp_bf16', False, None, False],
        [True, 'adamw', 'amp_bf16', False, None, False],
        [False, 'adam', 'amp_bf16', False, None, False],
        [False, 'adamw', 'amp_fp16', False, None, False],
        [False, 'adamw', 'amp_bf16', True, None, False],
        [False, 'adamw', 'amp_bf16', False, ['rng'], False],
    ],
)
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:.*metrics are not saved with sharded state dict.*:UserWarning')
@pytest.mark.filterwarnings(r'ignore:Please use DTensor instead and we are deprecating ShardedTensor.:UserWarning')
def test_fsdp_partitioned_state_dict_load(
    world_size,
    tmp_path: pathlib.Path,
    autoresume: bool,
    precision: str,
    optimizer: str,
    weights_only: bool,
    load_ignore_keys: Union[list[str], None],
    use_symlink: bool,
    use_remote,
    s3_bucket,
    s3_ephemeral_prefix,
    request,
):
    if weights_only and autoresume:
        pytest.xfail('Weights only with autoresume is not supported')
    load_ignore_keys = [] if load_ignore_keys is None else load_ignore_keys

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

    fsdp_config = FSDPConfig(state_dict_type='sharded')

    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        optimizer=optimizer,
        max_duration='2ba',
        save_interval='2ba',
        save_weights_only=weights_only,
        fsdp_config=fsdp_config,
    )
    run_name = trainer1.state.run_name
    trainer1.fit()
    rng1 = get_rng_state()
    state_dict_from_trainer1_ba2 = trainer1.state.state_dict()
    trainer1.close()

    if use_remote:
        load_path = 's3://' + save_folder.strip('s3://').format(
            run_name=run_name,
        ) + ('/ba2' if not use_symlink else '/latest-rank0.pt.symlink')
        object_store = S3ObjectStore(bucket=f'{s3_bucket}')
    else:
        object_store = None
        load_path = str(save_folder.format(run_name=run_name) / pathlib.Path('ba2'))

    assert not is_checkpoint_legacy_sharded(
        object_store=object_store,
        source_path=load_path.replace(f's3://{s3_bucket}/', ''),
    )

    if autoresume:
        load_path = None
    trainer2 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        load_path=load_path,
        precision=precision,
        autoresume=autoresume,
        run_name=run_name,
        max_duration='4ba',
        save_interval='4ba',
        optimizer=optimizer,
        load_weights_only=weights_only,
        fsdp_config=fsdp_config,
        load_ignore_keys=load_ignore_keys,
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()
    rng2 = trainer2._rng_state
    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(
        state_dict_from_trainer1_ba2,
        state_dict_from_trainer2,
    )
    if not weights_only:
        if any('rng' in x for x in load_ignore_keys):
            assert rng1 is not None and rng2 is None
        else:
            _compare_rng_states_between_trainers(rng1, rng2)
        _compare_optims_between_state_dicts(
            state_dict_from_trainer1_ba2,
            state_dict_from_trainer2,
        )
        _compare_metrics_between_state_dicts(
            state_dict_from_trainer1_ba2,
            state_dict_from_trainer2,
        )
        _compare_timestamps_between_state_dicts(
            state_dict_from_trainer1_ba2,
            state_dict_from_trainer2,
        )

    trainer2.fit()
    trainer2.close()


@pytest.mark.gpu
@pytest.mark.remote
@world_size(2)
@pytest.mark.parametrize('precision', ['amp_bf16', 'amp_fp16'])
@pytest.mark.parametrize('autoresume', [False, True])
@pytest.mark.parametrize('num_shards', [2, 4, 7])
@pytest.mark.parametrize('sharding_strategy', ['FULL_SHARD', 'SHARD_GRAD_OP'])
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:MosaicMLLogger is not in the state_dict.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:.*metrics are not saved with sharded state dict.*:UserWarning')
def test_elastic_resumption(
    world_size,
    tmp_path: pathlib.Path,
    autoresume: bool,
    precision: str,
    sharding_strategy,
    s3_bucket,
    s3_read_only_prefix,
    num_shards: int,
):
    if autoresume:
        run_name = 'my-autoresume-run'
    else:
        run_name = None

    base_path = (
        f's3://{s3_bucket}/{s3_read_only_prefix}/elastic_test/'
        f'{sharding_strategy.lower()}_sharded_{precision}_'
        f'{num_shards}/'
    )

    mono_load_path = os.path.join(base_path, 'mono.pt')
    mono_trainer = get_trainer(
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
        assert not is_checkpoint_legacy_sharded(
            object_store=S3ObjectStore(bucket=f'{s3_bucket}'),
            source_path=sharded_load_path.replace(f's3://{s3_bucket}/', ''),
        )

    sharded_trainer = get_trainer(
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
        load_weights_only=False,
        fsdp_config=FSDPConfig(state_dict_type='sharded'),
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

    def compare_state_dicts():
        state_dict_from_trainer1 = mono_trainer.state.state_dict()
        state_dict_from_trainer2 = get_mono_state_dict_from_sharded_one(sharded_trainer)
        # We are comparing full state dicts (all optim and model parameters are gathered on only rank 0)
        # so we only need to compare on rank 0. Comparing on other ranks may cause errors because some state_dicts will be empty.
        if dist.get_global_rank() == 0:
            _compare_model_params_between_state_dicts(
                state_dict_from_trainer1,
                state_dict_from_trainer2,
            )
            _compare_optims_between_state_dicts(
                state_dict_from_trainer1,
                state_dict_from_trainer2,
            )
            # Metrics are NOT equal as sharded checkpoints do not save or load metrics
            # _compare_metrics_between_state_dicts(state_dict_from_trainer1, state_dict_from_trainer2)
            _compare_timestamps_between_state_dicts(
                state_dict_from_trainer1,
                state_dict_from_trainer2,
            )

    # Compare state dicts.
    compare_state_dicts()


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('num_ckpts_to_keep', [-1, 1, 2, 3])
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:.*metrics are not saved with sharded state dict.*:UserWarning')
@pytest.mark.filterwarnings(r'ignore:Please use DTensor instead and we are deprecating ShardedTensor.:UserWarning')
def test_cleanup_sharded_checkpoints(
    world_size,
    tmp_path: pathlib.Path,
    num_ckpts_to_keep: int,
    s3_bucket,
    s3_ephemeral_prefix,
    request,
):
    run_name = None
    batches_to_train = 3

    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = os.path.join(tmp_paths[0], 'checkpoints', '{run_name}')
    save_filename = 'rank{rank}.pt'

    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        run_name=run_name,
        max_duration=f'{batches_to_train}ba',
        save_interval='1ba',
        save_num_checkpoints_to_keep=num_ckpts_to_keep,
        fsdp_config=FSDPConfig(state_dict_type='sharded'),
    )
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
        file_list = {'.metadata', *[f'__{rank}_0.distcp' for rank in range(dist.get_world_size())]}
        assert set(os.listdir(full_path_ckpt_dir)) == file_list


@pytest.mark.gpu
@world_size(2)
@pytest.mark.parametrize('weights_only', [False, True])
@pytest.mark.parametrize('planner', [None, 'rename'])
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('2.0'), reason='requires PyTorch 2.0 or higher')
@pytest.mark.filterwarnings(r'ignore:TypedStorage is deprecated.:UserWarning')
@pytest.mark.filterwarnings(r'ignore:.*metrics are not saved with sharded state dict.*:UserWarning')
def test_fsdp_planner(
    world_size,
    tmp_path: pathlib.Path,
    weights_only: bool,
    planner: Optional[str],
):

    from torch.distributed.checkpoint._nested_dict import flatten_state_dict
    from torch.distributed.checkpoint._sharded_tensor_utils import _flatten_sharded_tensors
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata

    class RenameSavePlanner(DefaultSavePlanner):

        def set_up_planner(
            self,
            state_dict: STATE_DICT_TYPE,
            is_coordinator: bool,
        ) -> None:
            # suffix all keys with `foo_``
            state_dict['state']['model'] = {k + '_foo': v for k, v in state_dict['state']['model'].items()}

            super().set_up_planner(state_dict, is_coordinator)

    class RenameLoadPlanner(DefaultLoadPlanner):

        def set_up_planner(
            self,
            state_dict: STATE_DICT_TYPE,
            metadata: Metadata,
            is_coordinator: bool,
        ) -> None:
            if 'state' not in state_dict:
                super().set_up_planner(state_dict, metadata, is_coordinator)
                return

            self.original_state_dict = state_dict

            state_dict = dict(state_dict.items())
            state_dict['state'] = dict(state_dict['state'].items())
            state_dict['state']['model'] = {k + '_foo': v for k, v in state_dict['state']['model'].items()}

            if self.flatten_sharded_tensors:
                state_dict = _flatten_sharded_tensors(state_dict)

            if self.flatten_state_dict:
                state_dict, self.mappings = flatten_state_dict(state_dict)

            self.state_dict = state_dict
            self.metadata = metadata
            self.is_coordinator = is_coordinator

    save_planner = planner
    load_planner = planner
    if planner == 'rename':
        save_planner = RenameSavePlanner()
        load_planner = RenameLoadPlanner()

    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = os.path.join(tmp_paths[0], 'checkpoints', '{run_name}')

    fsdp_config = FSDPConfig(
        state_dict_type='sharded',
        load_planner=load_planner,
        save_planner=save_planner,
    )

    trainer1 = get_trainer(
        save_folder=str(save_folder),
        max_duration='2ba',
        save_interval='2ba',
        save_weights_only=weights_only,
        fsdp_config=fsdp_config,
    )
    run_name = trainer1.state.run_name
    trainer1.fit()
    rng1 = get_rng_state()
    state_dict_from_trainer1_ba2 = trainer1.state.state_dict()
    trainer1.close()

    load_path = str(save_folder.format(run_name=run_name) / pathlib.Path('ba2'))

    assert not is_checkpoint_legacy_sharded(
        object_store=None,
        source_path=load_path,
    )

    trainer2 = get_trainer(
        save_folder=str(save_folder),
        load_path=load_path,
        run_name=run_name,
        max_duration='4ba',
        save_interval='4ba',
        load_weights_only=weights_only,
        fsdp_config=fsdp_config,
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()
    rng2 = trainer2._rng_state
    # Compare saved state and loaded state for both ranks.
    _compare_model_params_between_state_dicts(
        state_dict_from_trainer1_ba2,
        state_dict_from_trainer2,
    )
    if not weights_only:
        _compare_rng_states_between_trainers(rng1, rng2)
        _compare_optims_between_state_dicts(
            state_dict_from_trainer1_ba2,
            state_dict_from_trainer2,
        )
        _compare_metrics_between_state_dicts(
            state_dict_from_trainer1_ba2,
            state_dict_from_trainer2,
        )
        _compare_timestamps_between_state_dicts(
            state_dict_from_trainer1_ba2,
            state_dict_from_trainer2,
        )

    trainer2.fit()
    trainer2.close()
