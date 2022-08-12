# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pathlib
import shutil
import tarfile
import tempfile
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
import torch.distributed

from composer.callbacks import CheckpointSaver
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.precision import Precision
from composer.core.time import Time, TimeUnit, ensure_time
from composer.datasets.dataset_hparams import DatasetHparams
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.loggers import ObjectStoreLogger
from composer.optim import CosineAnnealingScheduler
from composer.optim.optimizer_hparams_registry import AdamWHparams
from composer.trainer.devices import Device, DeviceGPU
from composer.trainer.devices.device_cpu import DeviceCPU
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist, is_tar
from composer.utils.checkpoint import glob_filter
from composer.utils.iter_helpers import ensure_tuple
from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from composer.utils.object_store.object_store_hparams import LibcloudObjectStoreHparams
from tests.common import (EventCounterCallback, configure_dataset_hparams_for_synthetic,
                          configure_model_hparams_for_synthetic, deep_compare, device)


class DummyStatefulCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.random_value = time.time_ns()

    def state_dict(self) -> Dict[str, Any]:
        return {
            'random_value': self.random_value,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.random_value = state['random_value']


def assert_weights_equivalent(original_trainer_hparams: TrainerHparams,
                              new_trainer_hparams: TrainerHparams,
                              overwrite_load_path=True,
                              save_overwrite=True) -> Tuple[Trainer, Trainer]:
    """
    Strategy: get the weights from a new trainer
    Then assert that they are equivalent to the weights from the original model.
    """

    # load_weights_only is False since the original Trainer is testing full checkpoint recovery
    if overwrite_load_path:
        original_trainer_hparams.load_path = new_trainer_hparams.load_path
    original_trainer_hparams.load_weights_only = False
    original_trainer_hparams.load_strict_model_weights = False
    original_trainer_hparams.save_overwrite = save_overwrite
    original_trainer = original_trainer_hparams.initialize_object()
    original_weights = original_trainer.state.model.parameters()

    new_trainer_hparams.save_overwrite = save_overwrite
    new_trainer = new_trainer_hparams.initialize_object()
    recovered_weights = new_trainer.state.model.parameters()

    for p1, p2 in zip(original_weights, recovered_weights):
        assert (p1.data == p2.data).all()

    return original_trainer, new_trainer


def _load_checkpoint(checkpoint_dir: str, filename: str):
    filename = filename.format(rank=0)
    if not is_tar(filename):
        return torch.load(filename, map_location='cpu')

    with tarfile.open(filename) as tarball:
        tarball.extractall(checkpoint_dir)
    states_path = os.path.join(checkpoint_dir, 'composer_states.pt')
    return torch.load(states_path, map_location='cpu')


def assert_checkpoints_equivalent(
    checkpoint_file_a: str,
    checkpoint_file_b: str,
) -> None:

    with tempfile.TemporaryDirectory() as tmp_path:
        a_checkpoint_dir = os.path.join(tmp_path, 'a')
        b_checkpoint_dir = os.path.join(tmp_path, 'b')

        checkpoint_a = _load_checkpoint(a_checkpoint_dir, checkpoint_file_a)
        checkpoint_b = _load_checkpoint(b_checkpoint_dir, checkpoint_file_b)

        # Remove the event counter callback, since the number of fit_start events will differ
        del checkpoint_a['state']['callbacks']['EventCounterCallback']
        del checkpoint_b['state']['callbacks']['EventCounterCallback']

        # Remove the wall clock time
        del checkpoint_a['state']['timestamp']['Timestamp']['total_wct']
        del checkpoint_a['state']['timestamp']['Timestamp']['epoch_wct']
        del checkpoint_a['state']['timestamp']['Timestamp']['batch_wct']
        del checkpoint_b['state']['timestamp']['Timestamp']['total_wct']
        del checkpoint_b['state']['timestamp']['Timestamp']['epoch_wct']
        del checkpoint_b['state']['timestamp']['Timestamp']['batch_wct']

        # Remove run_name, since it's a function of time
        del checkpoint_a['state']['run_name']
        del checkpoint_b['state']['run_name']

        deep_compare(checkpoint_a, checkpoint_b)

        if 'model' not in checkpoint_a['state']:
            assert 'optimizer' not in checkpoint_a['state']
            assert 'model' not in checkpoint_b['state']
            assert 'optimizer' not in checkpoint_b['state']
            # it is a deepspeed checkpoint
            # TODO manually compare the model and optimizer states


def get_two_epoch_composer_hparams(composer_trainer_hparams: TrainerHparams, checkpoint_folder: str):
    composer_trainer_hparams.grad_accum = 2
    composer_trainer_hparams.loggers = []
    composer_trainer_hparams.train_batch_size = 8
    composer_trainer_hparams.eval_batch_size = 16
    composer_trainer_hparams.max_duration = '2ep'
    composer_trainer_hparams.precision = Precision.FP32
    composer_trainer_hparams.callbacks = [DummyStatefulCallback(), EventCounterCallback()]
    composer_trainer_hparams.train_subset_num_batches = 5
    composer_trainer_hparams.save_folder = checkpoint_folder
    composer_trainer_hparams.save_filename = 'ep{epoch}.pt'
    composer_trainer_hparams.save_interval = '1ep'
    composer_trainer_hparams.seed = None
    composer_trainer_hparams.eval_interval = '1ba'
    return composer_trainer_hparams


@pytest.mark.parametrize(
    'remove_field_paths,filter_params',
    [
        [[['state', 'model', 'classifier', 'weights'], ['state', 'model', 'classifier', 'bias']],
         ['state/model/classifier/weights', 'state/model/classifier/bias']],
        [[
            ['state', 'model', 'classifier', 'weights'],
            ['state', 'model', 'classifier', 'bias'],
        ], ['state/model/classifier/*']],
        [
            [['state', 'timestep']],
            ['state/timestep'],
        ],
        [
            [['state', 'list_element', 0]],
            ['state/list_element/0'],
        ],
        [
            [['state', 'list_element', 0, 'nested_list_element']],
            ['state/list_element/0/nested_list_element'],
        ],
        [
            # Repeating the zero for the test case as removing a list index shifts everything
            [['state', 'list_element', 0], ['state', 'list_element', 0]],
            ['state/list_element/0', 'state/list_element/1'],
        ],
        [
            [
                ['state', 'model', 'classifier', 'weights'],
                ['state', 'model', 'layer1', 'weights'],
                ['state', 'model', 'layer2', 'weights'],
            ],
            ['state/model/*/weights'],
        ],
        [
            [['state', 'model', 'layer1', 'weights'], ['state', 'model', 'layer2', 'weights']],
            ['state/model/layer*/weights'],
        ],
    ],
)
def test_ignore_params(remove_field_paths: List[List[str]], filter_params: List[str]):
    # Set up base dictionary
    base_dict = {
        'state': {
            'run_name': 'my_first_run',
            'timestep': 7,
            'list_element': [{
                'nested_list_element': 'hello'
            }, 'world'],
            'model': {
                'layer1': {
                    'weights': 6,
                    'bias': 2
                },
                'layer2': {
                    'weights': 7,
                    'bias': 1
                },
                'classifier': {
                    'weights': 5,
                    'bias': 3
                }
            }
        },
        'rng': 0,
    }

    # Remove classifier params
    new_dict = copy.deepcopy(base_dict)
    for remove_field_path in remove_field_paths:
        temp_dict = base_dict
        for step in remove_field_path[:-1]:
            temp_dict = temp_dict[step]
        del temp_dict[remove_field_path[-1]]

    glob_filter(filter_params)(new_dict)
    assert base_dict == new_dict


@device('cpu', 'gpu')
def test_load_weights(
    device: str,
    composer_trainer_hparams: TrainerHparams,
):
    """strategy:
    - train two epochs. capture checkpoints after `save_interval` and ep2.
    - create a new trainer from the `save_interval` checkpoint, but with a new optimizer and scheduler.
    - assert that the model weights are the original model, even though the optimizer and scheduler are different.
    """
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    checkpoint_a_folder = 'first'
    final_checkpoint = 'ep2.pt'
    composer_trainer_hparams = get_two_epoch_composer_hparams(composer_trainer_hparams, checkpoint_a_folder)

    second_trainer_hparams = copy.deepcopy(composer_trainer_hparams)
    _test_checkpoint_trainer(composer_trainer_hparams)

    # Reduce the filepath to get the location on the rank zero process
    checkpoint_a_file_path = [os.path.join(os.path.abspath(checkpoint_a_folder), final_checkpoint)]
    dist.broadcast_object_list(checkpoint_a_file_path)

    # load only model weights
    second_trainer_hparams.load_path = checkpoint_a_file_path[0]
    second_trainer_hparams.load_weights_only = True
    second_trainer_hparams.load_strict_model_weights = True
    # setup a new optimizer
    second_trainer_hparams.optimizers = AdamWHparams()

    # setup a new LR scheduler
    assert isinstance(second_trainer_hparams.max_duration, str)
    second_trainer_hparams.schedulers = [CosineAnnealingScheduler(t_max=second_trainer_hparams.max_duration)]

    # ensure our new choice of scheduler is different than the original scheduler
    for idx in range(len(second_trainer_hparams.schedulers)):
        if idx < len(ensure_tuple(composer_trainer_hparams.schedulers)):
            assert second_trainer_hparams.schedulers[idx] != ensure_tuple(composer_trainer_hparams.schedulers)[idx]

    # pass in the two trainers, verify that the weights are the same
    assert_weights_equivalent(
        original_trainer_hparams=composer_trainer_hparams,
        new_trainer_hparams=second_trainer_hparams,
    )


@device('cpu', 'gpu')
@pytest.mark.parametrize('use_object_store,delete_local_checkpoint', [
    pytest.param(False, False),
    pytest.param(True, False),
    pytest.param(True, True),
])
def test_autoresume(
    device: str,
    composer_trainer_hparams: TrainerHparams,
    use_object_store: bool,
    delete_local_checkpoint: bool,
    tmp_path: pathlib.Path,
    use_procs: bool = False,
):
    """strategy:
    - train two epochs. capture checkpoints after `save_interval` and ep2.
    - create a new trainer with autoresume=True.
    - assert that the model weights are the original model even though load_path is not set.
    """
    del device  # unused. Set automatically
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    if use_object_store:
        pytest.importorskip('libcloud')

    checkpoint_a_folder = 'first'
    checkpoint_b_folder = 'second'
    middle_checkpoint = 'ep1.pt'
    final_checkpoint = 'ep2.pt'
    latest_checkpoint = composer_trainer_hparams.save_latest_filename.format(rank=dist.get_global_rank())
    composer_trainer_hparams = get_two_epoch_composer_hparams(composer_trainer_hparams, checkpoint_a_folder)
    composer_trainer_hparams.run_name = 'big-chungus'
    second_trainer_hparams = copy.deepcopy(composer_trainer_hparams)
    # Add object store logger
    if use_object_store:
        remote_dir = str(tmp_path / 'object_store')
        os.makedirs(remote_dir, exist_ok=True)
        for hparams in [composer_trainer_hparams, second_trainer_hparams]:
            object_store_logger = ObjectStoreLogger(
                object_store_cls=LibcloudObjectStore,
                object_store_kwargs={
                    'provider': 'local',
                    'container': '.',
                    'provider_kwargs': {
                        'key': remote_dir,
                    },
                },
                num_concurrent_uploads=1,
                use_procs=use_procs,
                upload_staging_folder=str(tmp_path / 'staging_folder'),
            )
            hparams.loggers = [object_store_logger]

    _test_checkpoint_trainer(composer_trainer_hparams)

    # Create checkpoint in seperate folder to load. Optionally delete original checkpoint by moving it.
    if delete_local_checkpoint:
        shutil.move(checkpoint_a_folder, checkpoint_b_folder)
    else:
        shutil.copytree(checkpoint_a_folder, checkpoint_b_folder, symlinks=True)
    # Recreate symlink in new folder
    new_latest_path = os.path.join(checkpoint_b_folder, latest_checkpoint)
    os.remove(new_latest_path)
    os.symlink(final_checkpoint, new_latest_path)

    # Original trainer loads from filesystem using load_path
    composer_trainer_hparams.load_path = os.path.join(checkpoint_b_folder, latest_checkpoint)

    # re-create the trainer from the YAML
    second_trainer_hparams.autoresume = True
    # This should be ignored with autoresume
    second_trainer_hparams.load_path = middle_checkpoint

    # pass in the two trainers, verify that the weights are the same and run_name is same
    trainer, second_trainer = assert_weights_equivalent(original_trainer_hparams=composer_trainer_hparams,
                                                        new_trainer_hparams=second_trainer_hparams,
                                                        overwrite_load_path=False,
                                                        save_overwrite=False)
    assert trainer.state.run_name == second_trainer.state.run_name


@device('cpu', 'gpu')
def test_different_run_names(
    device: Device,
    composer_trainer_hparams: TrainerHparams,
):
    """strategy:
    - train two epochs and save checkpoints
    - load checkpoint with two different hparams and verify run_names are different as
        run_name should only be loaded from checkpoint if using autoresume
    """
    del device
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    # Train original checkpoints
    checkpoint_a_folder = 'first'
    final_checkpoint = 'ep2.pt'
    composer_trainer_hparams = get_two_epoch_composer_hparams(composer_trainer_hparams, checkpoint_a_folder)
    composer_trainer_hparams.save_overwrite = True
    composer_trainer_hparams.load_weights_only = False
    composer_trainer_hparams.load_strict_model_weights = False
    _test_checkpoint_trainer(composer_trainer_hparams)
    composer_trainer_hparams.load_path = os.path.join(checkpoint_a_folder, final_checkpoint)

    # Create new trainer and change seed for new run_name generation
    second_trainer_hparams = copy.deepcopy(composer_trainer_hparams)
    second_trainer_hparams.seed = 2

    trainer_a = composer_trainer_hparams.initialize_object()
    trainer_b = second_trainer_hparams.initialize_object()

    assert trainer_a.state.run_name != trainer_b.state.run_name


@device('cpu', 'gpu')
@pytest.mark.parametrize('save_overwrite', [
    True,
    False,
])
def test_save_overwrite(
    device: Device,
    composer_trainer_hparams: TrainerHparams,
    save_overwrite: bool,
):
    """strategy:
    - train two epochs. capture checkpoints after `save_interval` and ep2.
    - create a new trainer from the `save_interval` checkpoint, but with a new optimizer and scheduler.
    - assert that the model weights are the original model, even though the optimizer and scheduler are different.
    """
    del device
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return

    checkpoint_a_folder = 'first'
    middle_checkpoint = 'ep1.pt'
    final_checkpoint = 'ep2.pt'
    composer_trainer_hparams = get_two_epoch_composer_hparams(composer_trainer_hparams, checkpoint_a_folder)
    composer_trainer_hparams.save_overwrite = save_overwrite
    middle_trainer_hparams = copy.deepcopy(composer_trainer_hparams)
    final_trainer_hparams = copy.deepcopy(composer_trainer_hparams)
    _test_checkpoint_trainer(composer_trainer_hparams)

    # re-create the trainers from the YAMLs and move filepaths to rank zero process
    middle_checkpoint_path = [os.path.join(os.path.abspath(checkpoint_a_folder), middle_checkpoint)]
    dist.broadcast_object_list(middle_checkpoint_path)
    final_checkpoint_path = [os.path.join(os.path.abspath(checkpoint_a_folder), final_checkpoint)]
    dist.broadcast_object_list(final_checkpoint_path)

    # load model from middle checkpoint
    middle_trainer_hparams.load_path = middle_checkpoint_path[0]
    if save_overwrite:
        # succesfully load if save_overwrite is True
        trainer = middle_trainer_hparams.initialize_object()
        # Train for a minimal amount of time
        trainer.fit(duration='1ba')
    else:
        # raise FileExistsError if save_overwrite is False
        with pytest.raises(FileExistsError):
            trainer = middle_trainer_hparams.initialize_object()
            # Train for a minimal amount of time
            trainer.fit(duration='1ba')

    # load model from last checkpoint, which should work regardless of save_overwrite
    final_trainer_hparams.load_path = final_checkpoint_path[0]
    trainer = final_trainer_hparams.initialize_object()
    trainer.fit(duration='1ba')


def test_checkpoint_with_object_store_logger(
    composer_trainer_hparams: TrainerHparams,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Train model while logging to object store.

    Load model from object store and ensure it's the same.
    """
    pytest.importorskip('libcloud')

    checkpoint_a_folder = 'first'
    final_checkpoint = 'ep2.pt'
    composer_trainer_hparams = get_two_epoch_composer_hparams(
        composer_trainer_hparams,
        checkpoint_a_folder,
    )

    # Train model and log to object store
    remote_dir = str(tmp_path / 'object_store')
    os.makedirs(remote_dir, exist_ok=True)
    provider = 'local'
    container = '.'
    monkeypatch.setenv('OBJECT_STORE_KEY', remote_dir)  # for the local option, the key is the path
    object_store_hparams = LibcloudObjectStoreHparams(
        provider=provider,
        container=container,
        key_environ='OBJECT_STORE_KEY',
    )
    run_name = 'electric-zebra'
    composer_trainer_hparams.run_name = run_name
    second_trainer_hparams_object_store = copy.deepcopy(composer_trainer_hparams)
    second_trainer_hparams_logger = copy.deepcopy(composer_trainer_hparams)
    for hparams in [composer_trainer_hparams, second_trainer_hparams_logger]:
        object_store_logger = ObjectStoreLogger(
            object_store_cls=LibcloudObjectStore,
            object_store_kwargs={
                'provider': provider,
                'container': container,
                'provider_kwargs': {
                    'key': remote_dir,
                },
            },
            num_concurrent_uploads=1,
            use_procs=False,
            upload_staging_folder=str(tmp_path / 'staging_folder'),
        )
        hparams.loggers = [object_store_logger]
        if hparams is second_trainer_hparams_logger:
            hparams.load_logger_destination = object_store_logger

    artifact_name = f'{run_name}/checkpoints/ep2-ba10-rank' + '{rank}'
    trainer = composer_trainer_hparams.initialize_object()
    trainer.fit()

    trainer.close()

    # Load model weights using object store
    checkpoint_a_file_path = [os.path.join(os.path.abspath(checkpoint_a_folder), final_checkpoint)]
    dist.broadcast_object_list(checkpoint_a_file_path)
    composer_trainer_hparams.load_path = checkpoint_a_file_path[0]

    second_trainer_hparams_object_store.load_path = artifact_name
    second_trainer_hparams_object_store.load_object_store = object_store_hparams
    second_trainer_hparams_object_store.load_weights_only = True
    second_trainer_hparams_object_store.load_strict_model_weights = True
    composer_trainer_hparams.loggers = []

    assert_weights_equivalent(
        original_trainer_hparams=composer_trainer_hparams,
        new_trainer_hparams=second_trainer_hparams_object_store,
        overwrite_load_path=False,
    )

    # Load model weights using object store logger
    checkpoint_a_file_path = [os.path.join(os.path.abspath(checkpoint_a_folder), final_checkpoint)]
    dist.broadcast_object_list(checkpoint_a_file_path)

    second_trainer_hparams_logger.load_path = artifact_name
    second_trainer_hparams_logger.load_weights_only = True
    second_trainer_hparams_logger.load_strict_model_weights = True
    composer_trainer_hparams.loggers = []

    assert_weights_equivalent(
        original_trainer_hparams=composer_trainer_hparams,
        new_trainer_hparams=second_trainer_hparams_logger,
        overwrite_load_path=False,
    )


@pytest.mark.parametrize('world_size', [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize('device,deepspeed_enabled,zero_stage', [
    pytest.param('cpu', False, None, id='cpu-ddp'),
    pytest.param('gpu', False, None, id='gpu-ddp', marks=pytest.mark.gpu),
    pytest.param('gpu', True, 0, id='deepspeed-zero0', marks=pytest.mark.gpu),
    pytest.param('gpu', True, 1, id='deepspeed-zero1', marks=pytest.mark.gpu),
    pytest.param('gpu', True, 2, id='deepspeed-zero2', marks=pytest.mark.gpu),
])
@pytest.mark.parametrize(
    'seed,save_interval,save_filename,resume_file,final_checkpoint',
    [
        [None, '1ep', 'ep{epoch}-rank{rank}.pt', 'ep1-rank{rank}.pt', 'latest-rank{rank}.pt'
        ],  # test randomized seed saving and symlinking
        [42, '1ep', 'ep{epoch}-rank{rank}.pt', 'ep1-rank{rank}.pt', 'ep2-rank{rank}.pt'],  # test save at epoch end
        [42, '1ep', 'ep{epoch}-rank{rank}.tgz', 'ep1-rank{rank}.tgz', 'ep2-rank{rank}.tgz'
        ],  # test tarball with compression
        [42, '2ba', 'ba{batch}-rank{rank}.pt', 'ba4-rank{rank}.pt', 'ba8-rank{rank}.pt'
        ],  # test save batch in partial epoch
        [42, '1ba', 'ba{batch}-rank{rank}.pt', 'ba5-rank{rank}.pt', 'ba8-rank{rank}.pt'
        ],  # test save batch at epoch end
        [42, '2ba', 'ba{batch}-rank{rank}.pt', 'ba6-rank{rank}.pt', 'ba8-rank{rank}.pt'
        ],  # test save batch after complete epoch
    ],
)
@pytest.mark.parametrize('model_name', [
    None,
    pytest.param('resnet50_synthetic', marks=pytest.mark.daily),
    pytest.param('gpt2_52m', marks=pytest.mark.daily),
])
def test_checkpoint(
    device: str,
    world_size: int,
    deepspeed_enabled: bool,
    zero_stage: Optional[int],
    composer_trainer_hparams: TrainerHparams,
    save_interval: str,
    save_filename: str,
    resume_file: str,
    final_checkpoint: str,
    seed: Optional[int],
    model_name: Optional[str],
    tmp_path: pathlib.Path,
):
    """strategy:
    - train two epochs. capture checkpoints after `checkpoint_interval` and ep2.
    - create a new trainer from the `checkpoint_interval` checkpoint, and train until end. checkpoint again.
    - assert that the checkpoint from the new trainer at the end is the same as the checkpoint from the first trainer at the end.
    """
    del world_size  # unused. Read via env variable

    if deepspeed_enabled:
        if not is_tar(resume_file):
            resume_file += '.tar'
        if not is_tar(final_checkpoint):
            final_checkpoint += '.tar'

    if model_name is not None:
        if device == 'cpu':
            pytest.skip('Real models require a GPU -- otherwise they take too long')
        model_hparams = TrainerHparams.load(model_name)
        composer_trainer_hparams.train_dataset = model_hparams.train_dataset
        composer_trainer_hparams.val_dataset = model_hparams.val_dataset
        composer_trainer_hparams.model = model_hparams.model
        composer_trainer_hparams.optimizers = model_hparams.optimizers
        composer_trainer_hparams.schedulers = model_hparams.schedulers

    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip('Checkpointing tests require synthetic data')
        return

    configure_model_hparams_for_synthetic(composer_trainer_hparams.model)

    assert isinstance(composer_trainer_hparams.train_dataset, DatasetHparams)
    configure_dataset_hparams_for_synthetic(composer_trainer_hparams.train_dataset, composer_trainer_hparams.model)
    composer_trainer_hparams.save_filename = save_filename
    composer_trainer_hparams.train_dataset.shuffle = False

    assert isinstance(composer_trainer_hparams.val_dataset, DatasetHparams)
    configure_dataset_hparams_for_synthetic(composer_trainer_hparams.val_dataset, composer_trainer_hparams.model)
    composer_trainer_hparams.val_dataset.shuffle = False

    composer_trainer_hparams.grad_accum = 2
    composer_trainer_hparams.loggers = []
    composer_trainer_hparams.train_batch_size = 8
    composer_trainer_hparams.eval_batch_size = 16
    num_epochs = 2
    composer_trainer_hparams.max_duration = f'{num_epochs}ep'
    composer_trainer_hparams.precision = Precision.FP32
    composer_trainer_hparams.callbacks = [DummyStatefulCallback(), EventCounterCallback()]
    composer_trainer_hparams.train_subset_num_batches = 5
    composer_trainer_hparams.eval_subset_num_batches = 5
    composer_trainer_hparams.device = DeviceCPU() if device == 'cpu' else DeviceGPU()
    if deepspeed_enabled:
        assert zero_stage is not None
        if zero_stage > 0:
            if model_name is not None:
                pytest.skip(
                    textwrap.dedent(f"""\
                        Skipping test since deterministic mode is required for
                        non-trivial models, but deterministic mode isn't compatible with deepspeed
                        zero stage {zero_stage}"""))
        composer_trainer_hparams.deepspeed_config = {'zero_optimization': {'stage': zero_stage}}

    checkpoint_a_folder = str(tmp_path / 'first')
    composer_trainer_hparams.save_folder = checkpoint_a_folder
    composer_trainer_hparams.save_interval = save_interval
    composer_trainer_hparams.seed = seed

    if resume_file.startswith('ba'):
        composer_trainer_hparams.eval_interval = '1ba'
    if resume_file.startswith('ep'):
        composer_trainer_hparams.eval_interval = '1ep'

    second_trainer_hparams = copy.deepcopy(composer_trainer_hparams)
    first_trainer = _test_checkpoint_trainer(composer_trainer_hparams)
    dist.barrier()  # Ensure all ranks wrote the checkpoint file
    save_interval_time = Time.from_timestring(save_interval)
    if save_interval_time.unit == TimeUnit.EPOCH:
        expected_num_checkpoints = ((num_epochs - 1) // save_interval_time.value) + 1
    else:
        expected_num_checkpoints = (
            (composer_trainer_hparams.train_subset_num_batches * num_epochs - 1) // save_interval_time.value) + 1
    checkpoint_saver = None
    for callback in first_trainer.state.callbacks:
        if isinstance(callback, CheckpointSaver):
            checkpoint_saver = callback
    assert checkpoint_saver is not None
    assert len(checkpoint_saver.saved_checkpoints) == expected_num_checkpoints

    rank_to_checkpoint_a_folder = dist.all_gather_object(os.path.abspath(checkpoint_a_folder))

    checkpoint_to_resume_filepath = os.path.join(rank_to_checkpoint_a_folder[0], resume_file)
    first_trainer_final_checkpoint_filepath = os.path.join(rank_to_checkpoint_a_folder[0], final_checkpoint)

    # Move the resume and final file to the rank 0 folder
    try:
        rank_checkpoint_filepath = os.path.join(checkpoint_a_folder, resume_file.format(rank=dist.get_global_rank()))
        shutil.copy2(rank_checkpoint_filepath,
                     checkpoint_to_resume_filepath.format(rank=dist.get_global_rank()),
                     follow_symlinks=True)
    except (shutil.SameFileError, FileNotFoundError):
        pass

    try:
        rank_checkpoint_filepath = os.path.join(checkpoint_a_folder,
                                                final_checkpoint.format(rank=dist.get_global_rank()))
        shutil.copy2(rank_checkpoint_filepath,
                     first_trainer_final_checkpoint_filepath.format(rank=dist.get_global_rank()),
                     follow_symlinks=True)
    except (shutil.SameFileError, FileNotFoundError):
        pass

    checkpoint_b_folder = os.path.join(rank_to_checkpoint_a_folder[0], 'second')

    second_trainer_hparams.save_folder = checkpoint_b_folder
    second_trainer_hparams.load_path = checkpoint_to_resume_filepath
    second_trainer_hparams.load_weights_only = False
    second_trainer_hparams.load_strict_model_weights = False

    _test_checkpoint_trainer(second_trainer_hparams)
    dist.barrier()  # Ensure all ranks wrote the checkpoint file
    second_trainer_final_checkpoint_filepath = os.path.join(checkpoint_b_folder, final_checkpoint)

    assert_checkpoints_equivalent(
        checkpoint_file_a=first_trainer_final_checkpoint_filepath,
        checkpoint_file_b=second_trainer_final_checkpoint_filepath,
    )


def _test_checkpoint_trainer(trainer_hparams: TrainerHparams):
    trainer = trainer_hparams.initialize_object()
    trainer.fit()
    _validate_events_called_expected_number_of_times(trainer, ensure_time(trainer_hparams.eval_interval,
                                                                          TimeUnit.EPOCH))
    return trainer


def _validate_events_called_expected_number_of_times(trainer: Trainer, eval_interval: Time):
    state = trainer.state
    assert state.dataloader_label == 'train'
    assert state.dataloader_len is not None
    assert state.max_duration is not None
    assert state.max_duration.unit == TimeUnit.EPOCH
    num_epochs = state.max_duration.value
    num_total_steps = num_epochs * int(state.dataloader_len)
    num_total_microbatches = num_total_steps * state.grad_accum
    num_evals = 0
    if eval_interval.unit == TimeUnit.BATCH:
        num_evals = num_total_steps // int(eval_interval)
    if eval_interval.unit == TimeUnit.EPOCH:
        num_evals = num_epochs // int(eval_interval)

    assert trainer.state.evaluators is not None
    for evaluator in trainer.state.evaluators:
        assert evaluator.dataloader is not None
    assert trainer.state.evaluators[0].subset_num_batches != -1
    assert trainer.state.evaluators[0].subset_num_batches is not None
    num_eval_steps = num_evals * trainer.state.evaluators[0].subset_num_batches * len(trainer.state.evaluators)

    event_to_num_expected_invocations = {
        Event.INIT: 1,
        Event.EPOCH_START: num_epochs,
        Event.BATCH_START: num_total_steps,
        Event.AFTER_DATALOADER: num_total_steps,
        Event.BEFORE_FORWARD: num_total_microbatches,
        Event.AFTER_FORWARD: num_total_microbatches,
        Event.BEFORE_LOSS: num_total_microbatches,
        Event.AFTER_LOSS: num_total_microbatches,
        Event.BEFORE_BACKWARD: num_total_microbatches,
        Event.AFTER_BACKWARD: num_total_microbatches,
        Event.BEFORE_TRAIN_BATCH: num_total_steps,
        Event.AFTER_TRAIN_BATCH: num_total_steps,
        Event.BATCH_END: num_total_steps,
        Event.BATCH_CHECKPOINT: num_total_steps,
        Event.EPOCH_END: num_epochs,
        Event.EPOCH_CHECKPOINT: num_epochs,
        Event.EVAL_START: num_evals,
        Event.EVAL_BATCH_START: num_eval_steps,
        Event.EVAL_BEFORE_FORWARD: num_eval_steps,
        Event.EVAL_AFTER_FORWARD: num_eval_steps,
        Event.EVAL_BATCH_END: num_eval_steps,
        Event.EVAL_END: num_evals,
    }

    for callback in trainer.state.callbacks:
        if isinstance(callback, EventCounterCallback):
            for event, expected in event_to_num_expected_invocations.items():
                actual = callback.event_to_num_calls[event]
                assert expected == actual, f'Event {event} expected to be called {expected} times, but instead it was called {actual} times'
            return
    assert False, 'EventCounterCallback not found in callbacks'
