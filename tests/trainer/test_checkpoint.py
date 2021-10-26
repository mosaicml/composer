# Copyright 2021 MosaicML. All Rights Reserved.

import functools
import os
import random
import shutil
from logging import Logger
from typing import Dict, Optional

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

from composer.callbacks.callback_hparams import CallbackHparams
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.state import State
from composer.core.types import StateDict
from composer.trainer.ddp import FileStoreHparams
from composer.trainer.devices import CPUDeviceHparams, DeviceHparams, GPUDeviceHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams, callback_registry
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
            setattr(self, event.value, functools.partial(self._event_catchall, event=event))

    def _event_catchall(self, state: State, logger: Logger, event: Event):
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


@pytest.fixture
def checkpoint_folder(ddp_tmpdir) -> str:
    return os.path.join(ddp_tmpdir, "checkpoints")


@pytest.fixture
def checkpointing_trainer_hparams(mosaic_trainer_hparams: TrainerHparams, ddp_tmpdir: str) -> TrainerHparams:
    mosaic_trainer_hparams.grad_accum = 2
    mosaic_trainer_hparams.checkpoint_interval = 1
    mosaic_trainer_hparams.checkpoint_folder = os.path.join(ddp_tmpdir, "checkpoints")
    mosaic_trainer_hparams.max_epochs = 2
    mosaic_trainer_hparams.ddp.fork_rank_0 = False
    mosaic_trainer_hparams.callbacks.append(DummyStatefulCallbackHparams())
    mosaic_trainer_hparams.callbacks.append(EventCounterCallbackHparams())
    return mosaic_trainer_hparams


is_main_pytest_process: Optional[bool] = None


def _should_create_trainer(trainer_key: str) -> bool:
    # trainer_key is used for DDP forking, where a single test may create multiple trainers
    # each trainer should have a unique key. This key is stored in an environ, so when the forked process is created,
    # the test knows whether it got to the same point in the test as the main process.
    #
    # if the trainer key differs -- i.e. the test is forking for the 2nd trainer in a test, but the forked process
    # just got to the first trainer, or a forked process from the first trainer got to the point where it was going
    # to create the 2nd trainer, then _should_create_trainer returns False
    # otherwise, it returns True
    global is_main_pytest_process
    if is_main_pytest_process is None:
        is_main_pytest_process = "TRAINER_KEY" not in os.environ

    if is_main_pytest_process:
        os.environ["TRAINER_KEY"] = trainer_key
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("WORLD_SIZE", None)
    return os.environ["TRAINER_KEY"] == trainer_key


def get_trainer(device_hparams: DeviceHparams,
                trainer_hparams: TrainerHparams,
                *,
                trainer_key: str,
                ddp_tmpdir: str,
                checkpoint_filepath: Optional[str] = None) -> Optional[Trainer]:
    trainer_hparams.device = device_hparams
    trainer_hparams.ddp.store = FileStoreHparams(file_name=os.path.join(ddp_tmpdir, f"store_{trainer_key}"))
    trainer_hparams.checkpoint_filepath = checkpoint_filepath
    if not _should_create_trainer(trainer_key):
        return
    return Trainer.create_from_hparams(hparams=trainer_hparams)


def move_checkpoint(checkpoint_folder: str, name: str, destination_path: str) -> None:
    global is_main_pytest_process
    if not is_main_pytest_process:
        return
    checkpoint_filepath = os.path.join(checkpoint_folder, name)
    os.rename(checkpoint_filepath, destination_path)


def move_hparams(checkpoint_folder: str, destination_path: str) -> None:
    global is_main_pytest_process
    if not is_main_pytest_process:
        return
    checkpoint_filepath = os.path.join(checkpoint_folder, "hparams.yaml")
    os.rename(checkpoint_filepath, destination_path)


def assert_checkpoints_equivalent(hparams_file_a: str, checkpoint_file_a: str, hparams_file_b: str,
                                  checkpoint_file_b: str) -> None:
    checkpoint_a = torch.load(checkpoint_file_a, map_location='cpu')
    checkpoint_b = torch.load(checkpoint_file_b, map_location='cpu')

    hparams_a = TrainerHparams.create(hparams_file_a, cli_args=False)
    assert isinstance(hparams_a, TrainerHparams)
    hparams_b = TrainerHparams.create(hparams_file_b, cli_args=False)
    assert isinstance(hparams_b, TrainerHparams)

    # manually fix the store, as that part of the hparams WILL differ
    hparams_b.ddp.store = hparams_a.ddp.store

    hparams_a.checkpoint_filepath = hparams_b.checkpoint_filepath

    assert hparams_a.to_dict() == hparams_b.to_dict()

    hparams_a.checkpoint_filepath = checkpoint_file_a
    hparams_b.checkpoint_filepath = checkpoint_file_b

    trainer_a = Trainer.create_from_hparams(hparams=hparams_a)
    trainer_b = Trainer.create_from_hparams(hparams=hparams_b)

    state_a = trainer_a.state
    state_b = trainer_b.state
    assert_state_equivalent(state_a, state_b, skip_transient_fields=True)

    deep_compare(checkpoint_a["rng"], checkpoint_b["rng"])


@pytest.fixture(autouse=True)
def inject_stateful_callback_hparams(monkeypatch: MonkeyPatch):
    monkeypatch.setitem(callback_registry, "dummy", DummyStatefulCallbackHparams)
    monkeypatch.setitem(callback_registry, "event_counter", EventCounterCallbackHparams)


def clear_checkpoint_folder(checkpoint_folder: str):
    global is_main_pytest_process
    if not is_main_pytest_process:
        return
    shutil.rmtree(checkpoint_folder, ignore_errors=True)


@pytest.mark.run_long
@pytest.mark.timeout(90)
@pytest.mark.parametrize("device_hparams", [
    pytest.param(CPUDeviceHparams(n_cpus=1), id="1cpu"),
    pytest.param(CPUDeviceHparams(n_cpus=2), id='2cpu'),
    pytest.param(GPUDeviceHparams(n_gpus=1), marks=pytest.mark.n_gpus(1), id="1gpu"),
    pytest.param(GPUDeviceHparams(n_gpus=2), marks=pytest.mark.n_gpus(2), id="2gpu"),
])
@pytest.mark.parametrize("checkpoint_filename", ["ep1", "it4", "it1", "it6"])
@pytest.mark.parametrize("validate_every_n_batches,validate_every_n_epochs", [
    (0, 1),
    (1, 0),
])
def test_checkpoint(
    device_hparams: DeviceHparams,
    checkpointing_trainer_hparams: TrainerHparams,
    ddp_tmpdir: str,
    checkpoint_folder: str,
    checkpoint_filename: str,
    validate_every_n_batches: int,
    validate_every_n_epochs: int,
):
    """strategy:
    - train two epochs. capture checkpoints after `checkpoint_interval` and ep2.
    - create a new trainer from the `checkpoint_interval` checkpoint, and train until end. checkpoint again.
    - assert that the checkpoint from the new trainer at the end is the same as the checkpoint from the first trainer at the end.
    """
    checkpointing_trainer_hparams.checkpoint_interval_unit = "ep" if checkpoint_filename.startswith("ep") else "it"
    checkpointing_trainer_hparams.validate_every_n_batches = validate_every_n_batches
    checkpointing_trainer_hparams.validate_every_n_epochs = validate_every_n_epochs
    first_trainer = get_trainer(
        device_hparams=device_hparams,
        trainer_hparams=checkpointing_trainer_hparams,
        trainer_key="first",
        ddp_tmpdir=ddp_tmpdir,
    )
    checkpoint_a_file_path = os.path.join(ddp_tmpdir, "checkpoint_a.pt")
    checkpoint_b_file_path = os.path.join(ddp_tmpdir, 'checkpoint_b.pt')
    trainer_1_hparams_filepath = os.path.join(ddp_tmpdir, "trainer_1_hparams.yaml")
    final_checkpoint = "ep2.pt" if checkpoint_filename.startswith("ep") else "it8.pt"
    if first_trainer is not None:
        first_trainer.fit()
        move_checkpoint(checkpoint_folder, f"{checkpoint_filename}.pt", checkpoint_a_file_path)
        move_checkpoint(checkpoint_folder, final_checkpoint, checkpoint_b_file_path)
        move_hparams(checkpoint_folder, trainer_1_hparams_filepath)
        clear_checkpoint_folder(checkpoint_folder)
        validate_events_called_expected_number_of_times(first_trainer)
    second_trainer_hparams = TrainerHparams.create(trainer_1_hparams_filepath, cli_args=False)
    assert isinstance(second_trainer_hparams, TrainerHparams)
    second_trainer = get_trainer(
        device_hparams=device_hparams,
        trainer_hparams=second_trainer_hparams,
        trainer_key="second",
        ddp_tmpdir=ddp_tmpdir,
        checkpoint_filepath=checkpoint_a_file_path,
    )
    checkpoint_c_file_path = os.path.join(ddp_tmpdir, "checkpoint_c.pt")
    trainer_2_hparams_filepath = os.path.join(ddp_tmpdir, "trainer_2_hparams.yaml")
    if second_trainer is not None:
        second_trainer.fit()
        move_checkpoint(checkpoint_folder, final_checkpoint, checkpoint_c_file_path)
        move_hparams(checkpoint_folder, trainer_2_hparams_filepath)
        clear_checkpoint_folder(checkpoint_folder)
        validate_events_called_expected_number_of_times(second_trainer)

    if is_main_pytest_process:
        assert_checkpoints_equivalent(
            hparams_file_a=trainer_1_hparams_filepath,
            checkpoint_file_a=checkpoint_b_file_path,
            hparams_file_b=trainer_2_hparams_filepath,
            checkpoint_file_b=checkpoint_c_file_path,
        )


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
    assert state.eval_dataloader is not None
    num_eval_steps = num_evals * len(state.eval_dataloader)

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

    for callback in trainer.engine.callbacks:
        if isinstance(callback, EventCounterCallback):
            for event, expected in event_to_num_expected_invocations.items():
                actual = callback.event_to_num_calls[event]
                assert expected == actual, f"Event {event} expected to be called {expected} times, but instead it was called {actual} times"
            return
    assert False, "EventCounterCallback not found in callbacks"
