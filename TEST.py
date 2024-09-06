# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os

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
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from composer.algorithms import EMA
from composer.core import Algorithm, Event, Precision, State, Time
from composer.core.state import fsdp_get_optim_state_dict, fsdp_state_dict_type_context
from composer.models import ComposerClassifier
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from composer.utils import FSDPConfig, TPConfig, dist, parse_uri
from composer.utils.checkpoint import dist_cp_load, is_checkpoint_legacy_sharded
from composer.utils.file_helpers import get_file
from composer.utils.object_store import S3ObjectStore
from composer.utils.reproducibility import get_rng_state
from tests.common import RandomClassificationDataset, deep_compare
from tests.common.markers import world_size
from tests.trainer.test_checkpoint import TestCheckpointResumption, _assert_checkpoints_equivalent

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from tests.trainer.test_fsdp_checkpoint import _compare_model_params_between_state_dicts, _compare_optims_between_state_dicts, _compare_metrics_between_state_dicts
from icecream import install
from icecream import ic

install()
ic.configureOutput(includeContext=True)

def test_1(use_tp: bool):
    from tests.trainer.test_fsdp_checkpoint import get_trainer

    tmp_path: pathlib.Path = 'tmp'
    autoresume: bool = False
    precision: str = 'amp_bf16'
    optimizer: str = 'adam'
    save_weights_only: bool =  False
    load_weights_only: bool = False
    load_monolith_rank0_only: bool = False
    use_hsdp: bool = False

    if use_hsdp and version.parse(torch.__version__) < version.parse('2.4.0'):
        pytest.xfail('HSDP requires torch 2.4.0 or later')
    if use_tp and version.parse(torch.__version__) < version.parse('2.4.0'):
        pytest.skip('TP has full state dict issues before PyTorch 2.4.')
    if autoresume:
        run_name = 'my-cool-autoresume-run'
    else:
        run_name = None
    save_folder = tmp_path
    save_filename = 'rank{rank}.pt'

    if use_hsdp:
        fsdp_config = FSDPConfig(
            sharding_strategy='HYBRID_SHARD',
            sharded_ckpt_prefix_dir='ba{batch}',
            data_parallel_shard_degree=2,
            data_parallel_replicate_degree=2,
            sync_module_states=True,
        )
    else:
        fsdp_config = FSDPConfig(
            sharded_ckpt_prefix_dir='ba{batch}',
            sync_module_states=load_monolith_rank0_only,
            load_monolith_rank0_only=load_monolith_rank0_only,
        )
    tp_config = None
    if use_tp:
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
        tp_config = {
            'tensor_parallel_degree': 2,
            'layer_plan': {
                'module.0': ColwiseParallel(),
                'module.2': RowwiseParallel(),
            },
        }

    trainer1 = get_trainer(
        save_folder=str(save_folder),
        save_filename=save_filename,
        run_name=run_name,
        precision=precision,
        autoresume=autoresume,
        optimizer=optimizer,
        fsdp_config=fsdp_config,
        tp_config=tp_config,
    )

    if use_tp:
        assert trainer1.state.tp_config is not None
        assert isinstance(trainer1.state.tp_config, TPConfig)

    ic('Before trainer 1 fit')
    print('Before trainer 1 fit')
    trainer1.fit()
    print('After trainer 1 fit')
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
        save_weights_only=save_weights_only,
        load_weights_only=load_weights_only,
        tp_config=tp_config,
    )
    state_dict_from_trainer2 = trainer2.state.state_dict()

    if dist.get_global_rank() == 0:
        _compare_model_params_between_state_dicts(
            state_dict_from_trainer1,
            state_dict_from_trainer2,
        )
        if not load_weights_only:
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


def test_2(use_tp: bool):
    from tests.trainer.test_fsdp_checkpoint import SimpleMLP

    fsdp_config = FSDPConfig(sharded_ckpt_prefix_dir='ba{batch}')
    tp_config = None
    if use_tp:
        tp_config = {
            'tensor_parallel_degree': 2,
            'layer_plan': {'module.0': ColwiseParallel(), 'module.2': RowwiseParallel()},
        }

    model_init_device: str = 'cpu'
    num_features: int = 4
    num_classes: int = 2
    max_duration: Optional[int | str | Time] = '2ba'
    save_interval: str | int | Time | Callable[[State, Event], bool] = '2ba'

    model = SimpleMLP(num_features=num_features, num_classes=num_classes)
    model.module.to(model_init_device)
    dataset = RandomClassificationDataset(shape=(num_features,), num_classes=num_classes, size=128)
    dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset), batch_size=8,)
    optim = torch.optim.Adam(params=model.parameters())

    parallelism_config: dict[str, Union[FSDPConfig, dict[str, Any]]] = {'fsdp': fsdp_config}
    if tp_config is not None:
        parallelism_config['tp'] = tp_config

    trainer1 = Trainer(
        model=model,
        optimizers=optim,
        train_dataloader=dataloader,
        parallelism_config=parallelism_config,
        save_folder='tmp',
        max_duration=max_duration,
        save_interval=save_interval,
        save_filename='rank{rank}.pt',
        precision='amp_bf16',
        progress_bar=False,
        log_to_console=False,
        save_latest_filename='latest-rank{rank}.pt',
    )

    if use_tp:
        assert trainer1.state.tp_config is not None
        assert isinstance(trainer1.state.tp_config, TPConfig)

    ic('Before trainer 1 fit')
    print('Before trainer 1 fit')
    trainer1.fit()
    print('After trainer 1 fit')

if __name__ == '__main__':
    test = test_2
    verbose = True

    if not verbose:
        ic.disable()
        os.environ['NCCL_DEBUG'] = 'WARN'
    if verbose:
        os.environ['NCCL_DEBUG'] = 'INFO'

    print('*'*70, '\nuse_tp=False\n', '*'*70)
    test(use_tp=False)
    print('*'*70, '\nDone\n', '*'*70)

    print('*'*70, '\nuse_tp=True\n', '*'*70)
    test(use_tp=True)
    print('*'*70, '\nDone\n', '*'*70)
