# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from torch.optim import Adam
from torch.utils.data import DataLoader

from composer.algorithms import EMA
from composer.callbacks import SpeedMonitor
from composer.core import Precision
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from composer.utils import (
    StringEnum,
    convert_flat_dict_to_nested_dict,
    convert_nested_dict_to_flat_dict,
    extract_hparams,
)
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


def test_convert_nested_dict_to_flat_dict():
    test_nested_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}

    expected_flat_dict = {'a': 1, 'b/c': 2, 'b/d': 3, 'e/f/g': 4}
    actual_flat_dict = convert_nested_dict_to_flat_dict(test_nested_dict)
    assert actual_flat_dict == expected_flat_dict


def test_convert_flat_dict_to_nested_dict():
    expected_nested_dict = {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': {'f': {'g': 4}}}

    test_flat_dict = {'a': 1, 'b/c': 2, 'b/d': 3, 'e/f/g': 4}
    actual_nested_dict = convert_flat_dict_to_nested_dict(test_flat_dict)
    assert actual_nested_dict == expected_nested_dict


def test_extract_hparams():

    class Foo:

        def __init__(self) -> None:
            self.g = 7

    class Bar:

        def __init__(self) -> None:
            self.local_hparams = {'m': 11}

    class Baz(StringEnum):
        A = 'abc'

    locals_dict = {
        'a': 1.5,
        'b': {
            'c': 2.5,
            'd': 3,
        },
        'e': [4, 5, 6.2],
        'f': Foo(),
        'p': Bar(),
        '_g': 7,
        'h': None,
        'i': True,
        'j': Baz.A,
    }

    expected_parsed_dict = {
        'a': 1.5,
        'b': {
            'c': 2.5,
            'd': 3,
        },
        'e': [4, 5, 6.2],
        'f': 'Foo',
        'p': {
            'Bar': {
                'm': 11,
            },
        },
        'h': None,
        'i': True,
        'j': 'abc',
    }

    parsed_dict = extract_hparams(locals_dict)
    assert parsed_dict == expected_parsed_dict


def test_extract_hparams_trainer():
    train_dl = DataLoader(RandomClassificationDataset(), batch_size=16)
    model = SimpleModel()
    optimizer = Adam(model.parameters(), eps=1e-3)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        device_train_microbatch_size=16,
        precision=Precision.FP32,
        optimizers=optimizer,
        auto_log_hparams=True,
        progress_bar=False,
        log_to_console=False,
        run_name='test',
        seed=3,
        algorithms=[EMA()],
        loggers=[InMemoryLogger()],
        callbacks=[SpeedMonitor()],
    )

    expected_hparams = {
        'model': 'SimpleModel',

        # Train Dataloader
        'train_dataloader': 'DataLoader',
        'train_dataloader_label': 'train',
        'train_subset_num_batches': -1,
        'spin_dataloaders': True,

        # Stopping Condition
        'max_duration': None,

        # Algorithms
        'algorithms': ['EMA'],

        # Engine Pass Registration
        'algorithm_passes': None,

        # Optimizers and Scheduling
        'optimizers': 'Adam',
        'schedulers': None,
        'scale_schedule_ratio': 1.0,
        'step_schedulers_every_batch': None,

        # Evaluators
        'eval_dataloader': None,
        'eval_interval': 1,
        'eval_subset_num_batches': -1,

        # Callbacks and Logging
        'callbacks': ['SpeedMonitor'],
        'loggers': ['InMemoryLogger'],
        'run_name': 'test',
        'progress_bar': False,
        'log_to_console': False,
        'console_stream': 'stderr',
        'console_log_interval': '1ba',
        'log_traces': False,
        'auto_log_hparams': True,

        # Compile
        'compile_config': None,
        'is_model_compiled': False,

        # Load Checkpoint
        'load_path': None,
        'load_object_store': None,
        'load_weights_only': False,
        'load_strict_model_weights': True,
        'load_progress_bar': True,
        'load_ignore_keys': None,
        'load_exclude_algorithms': None,

        # Save Checkpoint
        'save_folder': None,
        'save_filename': 'ep{epoch}-ba{batch}-rank{rank}.pt',
        'save_latest_filename': 'latest-rank{rank}.pt',
        'save_overwrite': False,
        'save_interval': '1ep',
        'save_weights_only': False,
        'save_ignore_keys': None,
        'save_num_checkpoints_to_keep': -1,
        'save_metrics': False,

        # Graceful Resumption
        'autoresume': False,

        # DeepSpeed
        'deepspeed_config': None,
        'fsdp_config': None,
        'fsdp_auto_wrap': True,
        'parallelism_config': None,

        # System/Numerics
        'device': 'DeviceCPU',
        'precision': 'fp32',
        'precision_config': None,
        'device_train_microbatch_size': 16,

        # Reproducibility
        'seed': 3,
        'deterministic_mode': False,

        # Distributed Training
        'dist_timeout': 300.0,
        'ddp_sync_strategy': None,

        # Profiling
        'profiler': None,

        # Python logging
        'python_log_level': None,
        'auto_microbatching': False,
        'rank_zero_seed': 3,
        'latest_remote_file_name': None,
        'num_optimizers': 1,
        'remote_ud_has_format_string': [False],
    }

    assert trainer.local_hparams == expected_hparams
