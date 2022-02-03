# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
import random
import tarfile
import tempfile
import textwrap
from logging import Logger
from typing import Dict, Optional

import pytest
import torch
import torch.distributed
from _pytest.monkeypatch import MonkeyPatch

from composer.callbacks.callback_hparams import CallbackHparams
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.precision import Precision
from composer.core.state import State
from composer.core.types import Logger, StateDict
from composer.datasets import SyntheticHparamsMixin
from composer.optim import AdamWHparams
from composer.optim.scheduler import ConstantLRHparams, CosineAnnealingLRHparams
from composer.trainer.checkpoint import CheckpointLoader
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry
from composer.utils import run_directory
from composer.utils.object_store import ObjectStoreProviderHparams
from tests.test_state import assert_state_equivalent
from tests.utils.deep_compare import deep_compare


class DummyStatefulCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.random_value = random.random()

    def state_dict(self) -> StateDict:
        return {
            "random_value": self.random_value,
        }

    def load_state_dict(self, state: StateDict) -> None:
        self.random_value = state["random_value"]


class DummyStatefulCallbackHparams(CallbackHparams):

    def initialize_object(self) -> DummyStatefulCallback:
        return DummyStatefulCallback()


class EventCounterCallback(Callback):

    def __init__(self) -> None:
        self.event_to_num_calls: Dict[Event, int] = {}

        for event in Event:
            self.event_to_num_calls[event] = 0

    def _run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.TRAINING_START:
            # ignoring training start as it is called once per startup
            # and the states otherwise won't match
            return
        self.event_to_num_calls[event] += 1

    def state_dict(self) -> StateDict:
        return {"events": self.event_to_num_calls}

    def load_state_dict(self, state: StateDict) -> None:
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
    original_trainer_hparams.load_path = new_trainer_hparams.load_path
    original_trainer_hparams.load_weights_only = False
    original_trainer_hparams.load_strict_model_weights = False

    original_trainer = original_trainer_hparams.initialize_object()
    original_weights = original_trainer.state.model.parameters()

    new_trainer = new_trainer_hparams.initialize_object()
    recovered_weights = new_trainer.state.model.parameters()

    for p1, p2 in zip(original_weights, recovered_weights):
        assert (p1.data.ne(p2.data).sum() == 0)


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


def assert_checkpoints_equivalent(hparams_a: TrainerHparams, checkpoint_file_a: str, hparams_b: TrainerHparams,
                                  checkpoint_file_b: str) -> None:

    with tempfile.TemporaryDirectory() as tmpdir:
        a_checkpoint_dir = os.path.join(tmpdir, 'a')
        with tarfile.open(checkpoint_file_a.format(RANK=0)) as tarball_a:
            tarball_a.extractall(a_checkpoint_dir)
        a_states_dir = os.path.join(a_checkpoint_dir, 'composer_states.pt')

        b_checkpoint_dir = os.path.join(tmpdir, 'b')
        with tarfile.open(checkpoint_file_b.format(RANK=0)) as tarball_b:
            tarball_b.extractall(b_checkpoint_dir)
        b_states_dir = os.path.join(b_checkpoint_dir, 'composer_states.pt')

        checkpoint_a = torch.load(a_states_dir, map_location='cpu')
        checkpoint_b = torch.load(b_states_dir, map_location='cpu')

        deep_compare(checkpoint_a["rng"], checkpoint_b["rng"])

    assert hparams_b.load_path is not None
    assert hparams_b.save_folder is not None
    hparams_a.load_path = hparams_b.load_path
    hparams_a.load_weights_only = False
    hparams_a.load_strict_model_weights = False
    hparams_a.save_folder = hparams_b.save_folder

    assert hparams_a.to_dict() == hparams_b.to_dict()

    hparams_a.load_path = checkpoint_file_a
    hparams_b.load_path = checkpoint_file_b

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
    composer_trainer_hparams.save_interval = "1ep"
    composer_trainer_hparams.seed = None
    composer_trainer_hparams.validate_every_n_batches = 1
    composer_trainer_hparams.validate_every_n_epochs = 0
    final_checkpoint = "ep2.tar"
    _test_checkpoint_trainer(composer_trainer_hparams)

    # re-create the trainer from the YAML
    second_trainer_hparams = TrainerHparams.create(data=composer_trainer_hparams.to_dict(), cli_args=False)

    checkpoint_a_file_path = os.path.join(run_directory.get_run_directory(), checkpoint_a_folder, final_checkpoint)

    # load only model weights
    second_trainer_hparams.load_path = checkpoint_a_file_path
    second_trainer_hparams.load_weights_only = True
    second_trainer_hparams.load_strict_model_weights = True
    # setup a new optimizer
    second_trainer_hparams.optimizer = AdamWHparams()

    # setup a new LR scheduler
    scheduler_options = [ConstantLRHparams(), CosineAnnealingLRHparams(T_max=second_trainer_hparams.max_duration)]
    second_trainer_hparams.schedulers = [random.choice(scheduler_options)]

    # ensure our new choice of scheduler is different than the original scheduler
    for idx in range(len(second_trainer_hparams.schedulers)):
        if idx < len(composer_trainer_hparams.schedulers):
            assert second_trainer_hparams.schedulers[idx] != composer_trainer_hparams.schedulers[idx]

    # pass in the two trainers, verify that the weights are the same
    assert_weights_equivalent(
        original_trainer_hparams=composer_trainer_hparams,
        new_trainer_hparams=second_trainer_hparams,
    )


@pytest.mark.timeout(90000)
@pytest.mark.parametrize("world_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.world_size(2)),
])
@pytest.mark.parametrize("device_hparams,deepspeed_enabled,zero_stage", [
    pytest.param(CPUDeviceHparams(), False, None, id="cpu-ddp"),
    pytest.param(GPUDeviceHparams(), False, None, id="gpu-ddp", marks=pytest.mark.gpu),
    pytest.param(GPUDeviceHparams(), True, 0, id="deepspeed-zero0", marks=pytest.mark.deepspeed),
    pytest.param(GPUDeviceHparams(), True, 1, id="deepspeed-zero1", marks=pytest.mark.deepspeed),
    pytest.param(GPUDeviceHparams(), True, 2, id="deepspeed-zero2", marks=pytest.mark.deepspeed),
])
@pytest.mark.parametrize(
    "seed,checkpoint_filename,compression",
    [[None, "ep1.tar", ""], [42, "ep1.tar", ""], [42, "ep1.tar.gz", "gzip"], [42, "it4.tar", ""], [42, "it6.tar", ""]])
@pytest.mark.parametrize("model_name", [None, "resnet50_synthetic", "gpt2_52m"])
def test_checkpoint(
    device_hparams: DeviceHparams,
    world_size: int,
    deepspeed_enabled: bool,
    zero_stage: Optional[int],
    composer_trainer_hparams: TrainerHparams,
    checkpoint_filename: str,
    compression: str,
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
                        non-trivial models, but deterministic mode isn't compatible with deepsped
                        zero stage {zero_stage}"""))
        composer_trainer_hparams.deepspeed = {"zero_stage": zero_stage}

    checkpoint_a_folder = "first"
    composer_trainer_hparams.save_folder = checkpoint_a_folder
    composer_trainer_hparams.save_interval = "1ep" if checkpoint_filename.startswith("ep") else "2ba"
    composer_trainer_hparams.save_compression = compression
    composer_trainer_hparams.seed = seed

    composer_trainer_hparams.validate_every_n_batches = 0 if checkpoint_filename.startswith("it") else 1
    composer_trainer_hparams.validate_every_n_epochs = 0 if checkpoint_filename.startswith("ep") else 1
    final_checkpoint = ("ep2" if checkpoint_filename.startswith("ep") else "it8") + ".tar" + (".gz"
                                                                                              if compression else "")
    _test_checkpoint_trainer(composer_trainer_hparams)
    checkpoint_a_file_path = os.path.join(checkpoint_a_folder, checkpoint_filename)
    checkpoint_b_file_path = os.path.join(run_directory.get_node_run_directory(), "rank_{RANK}", checkpoint_a_folder,
                                          final_checkpoint)

    second_trainer_hparams = TrainerHparams.create(data=composer_trainer_hparams.to_dict(), cli_args=False)
    checkpoint_b_folder = "second"

    second_trainer_hparams.save_folder = checkpoint_b_folder
    second_trainer_filepath = os.path.join(run_directory.get_node_run_directory(), "rank_{RANK}",
                                           checkpoint_a_file_path)
    second_trainer_hparams.load_path = second_trainer_filepath
    second_trainer_hparams.load_weights_only = False
    second_trainer_hparams.load_strict_model_weights = False

    _test_checkpoint_trainer(second_trainer_hparams)

    checkpoint_c_file_path = os.path.join(run_directory.get_node_run_directory(), "rank_{RANK}", checkpoint_b_folder,
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
    validate_events_called_expected_number_of_times(trainer)


def validate_events_called_expected_number_of_times(trainer: Trainer):
    state = trainer.state

    num_epochs = state.max_epochs
    num_total_steps = num_epochs * state.steps_per_epoch
    num_total_microbatches = num_total_steps * state.grad_accum
    num_evals = 0
    if trainer.validate_every_n_batches > 0:
        num_evals = num_total_steps // trainer.validate_every_n_batches
    if trainer.validate_every_n_epochs > 0:
        num_evals = num_epochs // trainer.validate_every_n_epochs

    assert state.evaluators is not None
    for evaluator in state.evaluators:
        assert evaluator.dataloader is not None
    assert trainer._eval_subset_num_batches is not None
    num_eval_steps = num_evals * trainer._eval_subset_num_batches * len(state.evaluators)

    event_to_num_expected_invocations = {
        Event.INIT: 1,
        # training start is being ignored, as it should be called once per startup
        Event.TRAINING_START: 0,
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
        Event.EPOCH_END: num_epochs,
        Event.EVAL_START: num_evals,
        Event.EVAL_BATCH_START: num_eval_steps,
        Event.EVAL_BEFORE_FORWARD: num_eval_steps,
        Event.EVAL_AFTER_FORWARD: num_eval_steps,
        Event.EVAL_BATCH_END: num_eval_steps,
        Event.EVAL_END: num_evals,
        Event.TRAINING_END: 1,
    }

    for callback in trainer.state.callbacks:
        if isinstance(callback, EventCounterCallback):
            for event, expected in event_to_num_expected_invocations.items():
                actual = callback.event_to_num_calls[event]
                assert expected == actual, f"Event {event} expected to be called {expected} times, but instead it was called {actual} times"
            return
    assert False, "EventCounterCallback not found in callbacks"


def test_checkpoint_load_uri(tmpdir: pathlib.Path):
    loader = CheckpointLoader("https://example.com")
    loader._retrieve_checkpoint(destination_filepath=str(tmpdir / "example"), rank=0, ignore_not_found_errors=False)
    with open(str(tmpdir / "example"), "r") as f:
        assert f.readline().startswith("<!doctype html>")


def test_checkpoint_load_object_uri(tmpdir: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    try:
        import libcloud
        del libcloud
    except ImportError:
        pytest.skip("Skipping test as libcloud is not installed")

    remote_dir = tmpdir / "remote_dir"
    os.makedirs(remote_dir)
    monkeypatch.setenv("OBJECT_STORE_KEY", str(remote_dir))  # for the local option, the key is the path
    provider = ObjectStoreProviderHparams(
        provider='local',
        key_environ="OBJECT_STORE_KEY",
        container=".",
    ).initialize_object()
    with open(str(remote_dir / "checkpoint.txt"), 'wb') as f:
        f.write(b"checkpoint1")
    loader = CheckpointLoader("checkpoint.txt", object_store=provider)

    loader._retrieve_checkpoint(destination_filepath=str(tmpdir / "example"), rank=0, ignore_not_found_errors=False)
    with open(str(tmpdir / "example"), "rb") as f:
        f.read() == b"checkpoint1"
