# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import copy
import functools
import os
import pathlib
import shutil
import tarfile
import tempfile
import textwrap
import time
from glob import glob
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader

from composer.callbacks import CheckpointSaver
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.precision import Precision
from composer.core.time import Time, TimeUnit, ensure_time
from composer.datasets.dataset_hparams import DatasetHparams
from composer.datasets.synthetic_hparams import SyntheticHparamsMixin
from composer.loggers import ObjectStoreLogger
from composer.optim import ExponentialScheduler
from composer.trainer.devices import DeviceCPU, DeviceGPU
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils import dist, is_tar
from composer.utils.checkpoint import glob_filter
from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from tests.common import (EventCounterCallback, RandomImageDataset, SimpleConvModel,
                          configure_dataset_hparams_for_synthetic, configure_model_hparams_for_synthetic, deep_compare,
                          device)


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


class TestCheckpointLoading:

    def _assert_weights_equivalent(self, m1: torch.nn.Module, m2: torch.nn.Module):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)

    def get_trainer(self, **kwargs):
        model = SimpleConvModel()
        optimizer = torch.optim.Adam(model.parameters())

        return Trainer(
            model=model,
            train_dataloader=DataLoader(
                dataset=RandomImageDataset(),
                batch_size=8,
            ),
            eval_dataloader=DataLoader(
                dataset=RandomImageDataset(),
                batch_size=16,
            ),
            grad_accum=2,
            precision='fp32',
            train_subset_num_batches=5,
            save_interval='1ep',
            eval_interval='1ba',
            save_filename='ep{epoch}.pt',
            max_duration='2ep',
            optimizers=optimizer,
            schedulers=ExponentialScheduler(gamma=0.9),
            **kwargs,
        )

    def get_logger(self, tmp_path: pathlib.Path):
        """Returns an object store logger that saves locally."""
        remote_dir = str(tmp_path / 'object_store')
        os.makedirs(remote_dir, exist_ok=True)

        return ObjectStoreLogger(
            object_store_cls=LibcloudObjectStore,
            object_store_kwargs={
                'provider': 'local',
                'container': '.',
                'provider_kwargs': {
                    'key': remote_dir,
                },
            },
            num_concurrent_uploads=1,
            use_procs=False,
            upload_staging_folder=str(tmp_path / 'staging_folder'),
        )

    @device('cpu', 'gpu')
    def test_load_weights(self, device):

        trainer_1 = self.get_trainer(save_folder='first', device=device)
        trainer_1.fit()

        last_checkpoint = os.path.join('first', 'ep2.pt')
        trainer_2 = self.get_trainer(
            load_path=last_checkpoint,
            load_weights_only=True,
            load_strict_model_weights=True,
        )

        # check weights loaded properly
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

    def test_load_weights_object_store(self, tmp_path):

        trainer_1 = self.get_trainer(
            save_folder='first',
            loggers=[self.get_logger(tmp_path)],
            run_name='electric-zebra',
        )
        trainer_1.fit()

        trainer_2 = self.get_trainer(
            loggers=[self.get_logger(tmp_path)],
            run_name='electric-zebra',
            load_path='electric-zebra/checkpoints/latest-rank0',
            load_object_store=self.get_logger(tmp_path),
        )

        # check weights loaded properly
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

    # composer_trainer_hparams.callbacks = [DummyStatefulCallback(), EventCounterCallback()]

    @device('cpu', 'gpu')
    @pytest.mark.parametrize('use_object_store', [True, False])
    @pytest.mark.parametrize('delete_local', [True, False])
    def test_autoresume(
        self,
        device: str,
        tmp_path: pathlib.Path,
        use_object_store: bool,
        delete_local: bool,
    ):
        if delete_local and not use_object_store:
            pytest.skip('Invalid test setting.')

        if use_object_store:
            pytest.importorskip('libcloud')

        trainer_1 = self.get_trainer(
            save_folder='first',
            device=device,
            run_name='big-chungus',
            loggers=[self.get_logger(tmp_path)] if use_object_store else [],
        )

        # trains the model, saving the checkpoint files
        trainer_1.fit()

        if delete_local:
            # delete files locally, forcing trainer to look in object store
            shutil.rmtree('first')

        trainer_2 = self.get_trainer(
            save_folder='first',
            device=device,
            run_name='big-chungus',
            autoresume=True,
            load_path='ignored.pt',  # this should be ignored
            loggers=[self.get_logger(tmp_path)] if use_object_store else [],
        )

        # TODO (mihir): should be checking the entire state
        # not just the model?
        self._assert_weights_equivalent(
            trainer_1.state.model,
            trainer_2.state.model,
        )

        assert trainer_1.state.run_name == trainer_2.state.run_name

    @device('cpu', 'gpu')
    @pytest.mark.parametrize('save_overwrite', [True, False])
    def test_save_overwrite(self, device, save_overwrite):

        trainer_1 = self.get_trainer(
            save_folder='first',
            device=device,
        )
        trainer_1.fit()

        if save_overwrite:
            ctx = contextlib.nullcontext
        else:
            ctx = functools.partial(pytest.raises, FileExistsError)

        with ctx():  # expect FileExistsError if save_overwrite=False
            trainer_2 = self.get_trainer(
                save_folder='first',
                save_overwrite=save_overwrite,
                load_path=os.path.join('first', 'ep1.pt'),
                device=device,
            )
            trainer_2.fit(duration='1ba')

        # loading from the last checkpoint should work regardless
        # of save_overwrite, as new checkpoints are later in time.
        trainer_3 = self.get_trainer(
            save_folder='first',
            save_overwrite=save_overwrite,
            load_path=os.path.join('first', 'ep2.pt'),
            device=device,
        )
        trainer_3.fit(duration='1ba')


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
    del world_size

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

    file_was_saved = dist.get_global_rank() == 0 or deepspeed_enabled
    if not file_was_saved:
        expected_num_checkpoints = 0

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
def test_rotate_checkpoints(
    world_size,
    device,
    deepspeed_enabled,
    zero_stage,
    tmp_path: pathlib.Path,
):
    num_keep = 5

    # all ranks use rank 0 folder
    tmp_paths = dist.all_gather_object(os.path.abspath(tmp_path))
    save_folder = tmp_paths[0]

    if deepspeed_enabled:
        deepseed_config = {'zero_optimization': {'stage': zero_stage}}
    else:
        deepseed_config = None

    trainer = Trainer(
        model=SimpleConvModel(),
        train_dataloader=DataLoader(dataset=RandomImageDataset()),
        save_folder=str(save_folder),
        save_filename='checkpoint_{rank}_{batch}.pt',
        save_interval='1ba',
        max_duration='10ba',
        save_num_checkpoints_to_keep=num_keep,
        device=device,
        deepspeed_config=deepseed_config,
    )

    trainer.fit()

    dist.barrier()  # ensure all checkpoints rotated across ranks

    # deepspeed saves 1 file per rank
    expected_num = num_keep if not deepspeed_enabled else num_keep * world_size

    files = glob(os.path.join(save_folder, 'checkpoint_*'))
    assert len(files) == expected_num

    dist.barrier()  # all ranks finish before cleaning up tmpdir
