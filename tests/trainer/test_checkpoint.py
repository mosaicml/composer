# Copyright 2021 MosaicML. All Rights Reserved.

import os
import tarfile
import tempfile
import textwrap
import time
from typing import Any, Dict, Optional

import pytest
import torch
import torch.distributed
from _pytest.monkeypatch import MonkeyPatch

from composer.callbacks.callback_hparams import CallbackHparams
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.precision import Precision
from composer.core.state import State
from composer.core.time import Time, TimeUnit
from composer.datasets import SyntheticHparamsMixin
from composer.loggers import Logger
from composer.optim import AdamWHparams, CosineAnnealingSchedulerHparams
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry
from composer.utils import run_directory
from composer.utils.checkpoint import _is_archive
from tests.test_state import assert_state_equivalent
from tests.utils.deep_compare import deep_compare


class DummyStatefulCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.random_value = time.time_ns()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "random_value": self.random_value,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.random_value = state["random_value"]


class DummyStatefulCallbackHparams(CallbackHparams):

    def initialize_object(self) -> DummyStatefulCallback:
        return DummyStatefulCallback()


class EventCounterCallback(Callback):

    def __init__(self) -> None:
        self.event_to_num_calls: Dict[Event, int] = {}

        for event in Event:
            self.event_to_num_calls[event] = 0

    def run_event(self, event: Event, state: State, logger: Logger):
        self.event_to_num_calls[event] += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"events": self.event_to_num_calls}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.event_to_num_calls.update(state["events"])


class EventCounterCallbackHparams(CallbackHparams):

    def initialize_object(self) -> EventCounterCallback:
        return EventCounterCallback()


def assert_weights_equivalent(original_trainer_hparams: TrainerHparams, new_trainer_hparams: TrainerHparams) -> None:
    """
    Strategy: get the weights from a new trainer
    Then assert that they are equivalent to the weights from the original model.
    """

    # load_weights_only is False since the original Trainer is testing full checkpoint recovery
    original_trainer_hparams.load_path_format = new_trainer_hparams.load_path_format
    original_trainer_hparams.load_weights_only = False
    original_trainer_hparams.load_strict_model_weights = False

    original_trainer = original_trainer_hparams.initialize_object()
    original_weights = original_trainer.state.model.parameters()

    new_trainer = new_trainer_hparams.initialize_object()
    recovered_weights = new_trainer.state.model.parameters()

    for p1, p2 in zip(original_weights, recovered_weights):
        assert (p1.data == p2.data).all()


@pytest.fixture
def checkpointing_trainer_hparams(composer_trainer_hparams: TrainerHparams) -> TrainerHparams:
    composer_trainer_hparams.grad_accum = 2
    composer_trainer_hparams.max_duration = "2ep"
    composer_trainer_hparams.save_folder = "checkpoints"
    composer_trainer_hparams.save_interval = "1ba"
    composer_trainer_hparams.callbacks.append(DummyStatefulCallbackHparams())
    composer_trainer_hparams.callbacks.append(EventCounterCallbackHparams())
    composer_trainer_hparams.train_subset_num_batches = 5
    return composer_trainer_hparams


def _load_checkpoint(checkpoint_dir: str, filename: str):
    filename = filename.format(rank=0)
    if not _is_archive(filename):
        return torch.load(filename, map_location='cpu')

    with tarfile.open(filename) as tarball:
        tarball.extractall(checkpoint_dir)
    states_path = os.path.join(checkpoint_dir, 'composer_states.pt')
    return torch.load(states_path, map_location='cpu')


def assert_checkpoints_equivalent(hparams_a: TrainerHparams, checkpoint_file_a: str, hparams_b: TrainerHparams,
                                  checkpoint_file_b: str) -> None:

    with tempfile.TemporaryDirectory() as tmpdir:
        a_checkpoint_dir = os.path.join(tmpdir, 'a')
        b_checkpoint_dir = os.path.join(tmpdir, 'b')

        checkpoint_a = _load_checkpoint(a_checkpoint_dir, checkpoint_file_a)
        checkpoint_b = _load_checkpoint(b_checkpoint_dir, checkpoint_file_b)

        deep_compare(checkpoint_a["rng"], checkpoint_b["rng"])

    assert hparams_b.load_path_format is not None
    assert hparams_b.save_folder is not None
    hparams_a.load_path_format = hparams_b.load_path_format
    hparams_a.load_weights_only = False
    hparams_a.load_strict_model_weights = False
    hparams_a.save_folder = hparams_b.save_folder

    assert hparams_a.to_dict() == hparams_b.to_dict()

    hparams_a.load_path_format = checkpoint_file_a
    hparams_b.load_path_format = checkpoint_file_b

    trainer_a = hparams_a.initialize_object()
    state_a = trainer_a.state

    trainer_b = hparams_b.initialize_object()
    state_b = trainer_b.state

    assert_state_equivalent(state_a, state_b)


@pytest.fixture(autouse=True)
def inject_stateful_callback_hparams(monkeypatch: MonkeyPatch):
    monkeypatch.setitem(callback_registry, "dummy", DummyStatefulCallbackHparams)
    monkeypatch.setitem(callback_registry, "event_counter", EventCounterCallbackHparams)


@pytest.mark.timeout(90)
@pytest.mark.parametrize("device_hparams", [
    pytest.param(CPUDeviceHparams(), id="cpu"),
    pytest.param(GPUDeviceHparams(), id="gpu", marks=pytest.mark.gpu),
])
def test_load_weights(
    device_hparams: DeviceHparams,
    composer_trainer_hparams: TrainerHparams,
):
    """strategy:
    - train two epochs. capture checkpoints after `checkpoint_interval` and ep2.
    - create a new trainer from the `checkpoint_interval` checkpoint, but with a new optimizer and scheduler.
    - assert that the model weights are the original model, even though the optimizer and scheduler are different.
    """
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")
        return
    composer_trainer_hparams.device = device_hparams
    composer_trainer_hparams.train_dataset.use_synthetic = True
    composer_trainer_hparams.train_dataset.shuffle = False
    composer_trainer_hparams.val_dataset.use_synthetic = True
    composer_trainer_hparams.val_dataset.shuffle = False
    composer_trainer_hparams.grad_accum = 2
    composer_trainer_hparams.loggers = []
    composer_trainer_hparams.train_batch_size = 8
    composer_trainer_hparams.eval_batch_size = 16
    composer_trainer_hparams.max_duration = "2ep"
    composer_trainer_hparams.precision = Precision.FP32
    composer_trainer_hparams.callbacks = [DummyStatefulCallbackHparams(), EventCounterCallbackHparams()]
    composer_trainer_hparams.train_subset_num_batches = 5
    composer_trainer_hparams.device = device_hparams
    checkpoint_a_folder = "first"
    composer_trainer_hparams.save_folder = checkpoint_a_folder
    composer_trainer_hparams.save_name_format = "ep{epoch}.pt"
    composer_trainer_hparams.save_interval = "1ep"
    composer_trainer_hparams.seed = None
    composer_trainer_hparams.validate_every_n_batches = 1
    composer_trainer_hparams.validate_every_n_epochs = 0
    final_checkpoint = "ep2.pt"
    _test_checkpoint_trainer(composer_trainer_hparams)

    # re-create the trainer from the YAML
    second_trainer_hparams = TrainerHparams.create(data=composer_trainer_hparams.to_dict(), cli_args=False)

    checkpoint_a_file_path = os.path.join(run_directory.get_run_directory(), checkpoint_a_folder, final_checkpoint)

    # load only model weights
    second_trainer_hparams.load_path_format = checkpoint_a_file_path
    second_trainer_hparams.load_weights_only = True
    second_trainer_hparams.load_strict_model_weights = True
    # setup a new optimizer
    second_trainer_hparams.optimizer = AdamWHparams()

    # setup a new LR scheduler
    second_trainer_hparams.schedulers = [CosineAnnealingSchedulerHparams(t_max=second_trainer_hparams.max_duration)]

    # ensure our new choice of scheduler is different than the original scheduler
    for idx in range(len(second_trainer_hparams.schedulers)):
        if idx < len(composer_trainer_hparams.schedulers):
            assert second_trainer_hparams.schedulers[idx] != composer_trainer_hparams.schedulers[idx]

    # pass in the two trainers, verify that the weights are the same
    assert_weights_equivalent(
        original_trainer_hparams=composer_trainer_hparams,
        new_trainer_hparams=second_trainer_hparams,
    )


@pytest.mark.timeout(180)
@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize("device_hparams,deepspeed_enabled,zero_stage", [
    pytest.param(CPUDeviceHparams(), False, None, id="cpu-ddp"),
    pytest.param(GPUDeviceHparams(), False, None, id="gpu-ddp", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 0, id="deepspeed-zero0", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 1, id="deepspeed-zero1", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 2, id="deepspeed-zero2", marks=pytest.mark.gpu),
])
@pytest.mark.parametrize(
    "seed,save_interval,save_name_format,resume_file,final_checkpoint",
    [
        [None, "1ep", "ep{epoch}", "ep1", "latest-rank{rank}"],  # test randomized seed saving and symlinking
        [42, "1ep", "ep{epoch}", "ep1", "ep2"],  # test save at epoch end
        [42, "1ep", "ep{epoch}.tgz", "ep1.tgz", "ep2.tgz"],  # test tarball with compression
        [42, "2ba", "ba{batch}", "ba4", "ba8"],  # test save batch in partial epoch
        [42, "1ba", "ba{batch}", "ba5", "ba8"],  # test save batch at epoch end
        [42, "2ba", "ba{batch}", "ba6", "ba8"],  # test save batch after complete epoch
    ],
)
@pytest.mark.parametrize("model_name", [None, "resnet50_synthetic", "gpt2_52m"])
def test_checkpoint(
    device_hparams: DeviceHparams,
    world_size: int,
    deepspeed_enabled: bool,
    zero_stage: Optional[int],
    composer_trainer_hparams: TrainerHparams,
    save_interval: str,
    save_name_format: str,
    resume_file: str,
    final_checkpoint: str,
    seed: Optional[int],
    model_name: Optional[str],
):
    """strategy:
    - train two epochs. capture checkpoints after `checkpoint_interval` and ep2.
    - create a new trainer from the `checkpoint_interval` checkpoint, and train until end. checkpoint again.
    - assert that the checkpoint from the new trainer at the end is the same as the checkpoint from the first trainer at the end.
    """
    del world_size  # unused. Read via env variable

    if not isinstance(device_hparams, GPUDeviceHparams) and deepspeed_enabled:
        pytest.skip("DeepSpeed tests must be ran on GPU")

    if deepspeed_enabled:
        if not _is_archive(resume_file):
            resume_file += ".tar"
        if not _is_archive(final_checkpoint):
            final_checkpoint += ".tar"

    if model_name is not None:
        if not isinstance(device_hparams, GPUDeviceHparams):
            pytest.skip("Real models require a GPU -- otherwise they take too long")
        model_hparams = TrainerHparams.load(model_name)
        composer_trainer_hparams.train_dataset = model_hparams.train_dataset
        composer_trainer_hparams.val_dataset = model_hparams.val_dataset
        composer_trainer_hparams.model = model_hparams.model
        composer_trainer_hparams.optimizer = model_hparams.optimizer
        composer_trainer_hparams.schedulers = model_hparams.schedulers
    if not isinstance(composer_trainer_hparams.train_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")
        return
    if not isinstance(composer_trainer_hparams.val_dataset, SyntheticHparamsMixin):
        pytest.skip("Checkpointing tests require synthetic data")
        return

    composer_trainer_hparams.save_name_format = save_name_format
    composer_trainer_hparams.train_dataset.use_synthetic = True
    composer_trainer_hparams.train_dataset.shuffle = False
    composer_trainer_hparams.val_dataset.use_synthetic = True
    composer_trainer_hparams.val_dataset.shuffle = False
    composer_trainer_hparams.grad_accum = 2
    composer_trainer_hparams.loggers = []
    composer_trainer_hparams.train_batch_size = 8
    composer_trainer_hparams.eval_batch_size = 16
    num_epochs = 2
    composer_trainer_hparams.max_duration = f"{num_epochs}ep"
    composer_trainer_hparams.precision = Precision.FP32
    composer_trainer_hparams.callbacks = [DummyStatefulCallbackHparams(), EventCounterCallbackHparams()]
    composer_trainer_hparams.train_subset_num_batches = 5
    composer_trainer_hparams.eval_subset_num_batches = 5
    composer_trainer_hparams.device = device_hparams
    if deepspeed_enabled:
        assert zero_stage is not None
        if zero_stage > 0:
            composer_trainer_hparams.deterministic_mode = False
            if model_name is not None:
                pytest.skip(
                    textwrap.dedent(f"""\
                        Skipping test since deterministic mode is required for
                        non-trivial models, but deterministic mode isn't compatible with deepspeed
                        zero stage {zero_stage}"""))
        composer_trainer_hparams.deepspeed = {"zero_optimization": {"stage": zero_stage}}

    checkpoint_a_folder = "first"
    composer_trainer_hparams.save_folder = checkpoint_a_folder
    composer_trainer_hparams.save_interval = save_interval
    composer_trainer_hparams.seed = seed

    composer_trainer_hparams.validate_every_n_batches = 1 if resume_file.startswith("ba") else 0
    composer_trainer_hparams.validate_every_n_epochs = 1 if resume_file.startswith("ep") else 0
    first_trainer = _test_checkpoint_trainer(composer_trainer_hparams)
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
    checkpoint_a_file_path = os.path.join(checkpoint_a_folder, resume_file)
    checkpoint_b_file_path = os.path.join(run_directory.get_node_run_directory(), "rank_{rank}", checkpoint_a_folder,
                                          final_checkpoint)

    second_trainer_hparams = TrainerHparams.create(data=composer_trainer_hparams.to_dict(), cli_args=False)
    checkpoint_b_folder = "second"

    second_trainer_hparams.save_folder = checkpoint_b_folder
    second_trainer_filepath = os.path.join(run_directory.get_node_run_directory(), "rank_{rank}",
                                           checkpoint_a_file_path)
    second_trainer_hparams.load_path_format = second_trainer_filepath
    second_trainer_hparams.load_weights_only = False
    second_trainer_hparams.load_strict_model_weights = False

    _test_checkpoint_trainer(second_trainer_hparams)
    checkpoint_c_file_path = os.path.join(run_directory.get_node_run_directory(), "rank_{rank}", checkpoint_b_folder,
                                          final_checkpoint)

    assert_checkpoints_equivalent(
        hparams_a=composer_trainer_hparams,
        checkpoint_file_a=checkpoint_b_file_path,
        hparams_b=second_trainer_hparams,
        checkpoint_file_b=checkpoint_c_file_path,
    )


def _test_checkpoint_trainer(trainer_hparams: TrainerHparams):
    callback_registry["dummy"] = DummyStatefulCallbackHparams
    callback_registry["event_counter"] = EventCounterCallbackHparams

    trainer = trainer_hparams.initialize_object()
    trainer.fit()
    _validate_events_called_expected_number_of_times(trainer)
    return trainer


def _validate_events_called_expected_number_of_times(trainer: Trainer):
    state = trainer.state

    assert state.max_duration.unit == TimeUnit.EPOCH
    num_epochs = state.max_duration.value
    num_total_steps = num_epochs * state.steps_per_epoch
    num_total_microbatches = num_total_steps * state.grad_accum
    num_evals = 0
    if trainer._validate_every_n_batches > 0:
        num_evals = num_total_steps // trainer._validate_every_n_batches
    if trainer._validate_every_n_epochs > 0:
        num_evals = num_epochs // trainer._validate_every_n_epochs

    assert state.evaluators is not None
    for evaluator in state.evaluators:
        assert evaluator.dataloader is not None
    assert trainer._eval_subset_num_batches is not None
    num_eval_steps = num_evals * trainer._eval_subset_num_batches * len(state.evaluators)

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
                assert expected == actual, f"Event {event} expected to be called {expected} times, but instead it was called {actual} times"
            return
    assert False, "EventCounterCallback not found in callbacks"
