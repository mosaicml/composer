# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
import shutil
import tarfile
import tempfile
import textwrap
import time
from typing import Any, Dict, Optional, Union

import pytest
import torch
import torch.distributed
from torch.utils.data import DataLoader

from composer.core import DataSpec
from composer.core.callback import Callback
from composer.core.event import Event
from composer.core.state import State
from composer.loggers import Logger
from composer.trainer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


class EventCounterCallback(Callback):

    def __init__(self) -> None:
        self.event_to_num_calls: Dict[Event, int] = {}

        for event in Event:
            self.event_to_num_calls[event] = 0

    def run_event(self, event: Event, state: State, logger: Logger):
        del state, logger  # unused
        self.event_to_num_calls[event] += 1

    def state_dict(self) -> Dict[str, Any]:
        return {"events": self.event_to_num_calls}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.event_to_num_calls.update(state["events"])


def _predict_events_called_expected_number_of_times(trainer: Trainer, length):

    num_predicts = 1
    num_predict_steps = length

    event_to_num_expected_invocations = {
        Event.PREDICT_START: num_predicts,
        Event.PREDICT_BATCH_START: num_predict_steps,
        Event.PREDICT_BEFORE_FORWARD: num_predict_steps,
        Event.PREDICT_AFTER_FORWARD: num_predict_steps,
        Event.PREDICT_BATCH_END: num_predict_steps,
        Event.PREDICT_END: num_predicts,
    }
    for callback in trainer.state.callbacks:
        if isinstance(callback, EventCounterCallback):
            for event, expected in event_to_num_expected_invocations.items():
                actual = callback.event_to_num_calls[event]
                assert expected == actual, f"Event {event} expected to be called {expected} times, but instead it was called {actual} times"
            return
    assert False, "EventCounterCallback not found in callbacks"


class TestTrainerPredict():

    @pytest.fixture
    def config(self):
        return {
            'model': SimpleModel(),
            'train_dataloader': DataLoader(dataset=RandomClassificationDataset()),
            'eval_dataloader': DataLoader(dataset=RandomClassificationDataset()),
            'max_duration': '2ep',
            'callbacks': [EventCounterCallback()],
        }

    def test_predict(self, config):
        trainer = Trainer(**config)
        trainer.fit()
        trainer.predict(config['eval_dataloader'])
        _predict_events_called_expected_number_of_times(trainer, len(config['eval_dataloader']))
