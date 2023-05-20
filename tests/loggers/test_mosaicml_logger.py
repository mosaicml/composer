# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Type

import mcli
import pytest
import torch
from torch.utils.data import DataLoader

from composer.core import Callback
from composer.loggers import WandBLogger
from composer.loggers.mosaicml_logger import MosaicMLLogger, format_data_to_json_serializable
from composer.trainer import Trainer
from composer.utils import dist
from tests.callbacks.callback_settings import get_cb_kwargs, get_cbs_and_marks
from tests.common import RandomClassificationDataset, SimpleModel
from tests.common.markers import world_size


class MockMAPI:

    def __init__(self):
        self.run_metadata = {}

    def update_run_metadata(self, run_name, new_metadata):
        if run_name not in self.run_metadata:
            self.run_metadata[run_name] = {}
        for k, v in new_metadata.items():
            self.run_metadata[run_name][k] = v
        # Serialize the data to ensure it is json serializable
        json.dumps(self.run_metadata[run_name])


def test_format_data_to_json_serializable():
    data = {
        'key1': 'value1',
        'key2': 42,
        'key3': 3.14,
        'key4': True,
        'key5': torch.tensor([1, 2, 3]),
        'key6': {
            'inner_key': 'inner_value'
        },
        'key7': [1, 2, 3],
    }
    formatted_data = format_data_to_json_serializable(data)

    expected_formatted_data = {
        'key1': 'value1',
        'key2': 42,
        'key3': 3.14,
        'key4': True,
        'key5': 'Tensor of shape torch.Size([3])',
        'key6': {
            'inner_key': 'inner_value'
        },
        'key7': [1, 2, 3],
    }

    assert formatted_data == expected_formatted_data


@pytest.mark.parametrize('callback_cls', get_cbs_and_marks(callbacks=True))
@world_size(1, 2)
def test_logged_data_is_json_serializable(monkeypatch, callback_cls: Type[Callback], world_size):
    """Test that all logged data is json serializable, which is a requirement to use MAPI."""

    mock_mapi = MockMAPI()
    monkeypatch.setattr(mcli, 'update_run_metadata', mock_mapi.update_run_metadata)
    run_name = 'small_chungus'
    monkeypatch.setenv('RUN_NAME', run_name)

    callback_kwargs = get_cb_kwargs(callback_cls)
    callback = callback_cls(**callback_kwargs)
    train_dataset = RandomClassificationDataset()
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(
            train_dataset,
            sampler=dist.get_sampler(train_dataset),
        ),
        train_subset_num_batches=2,
        max_duration='1ep',
        callbacks=callback,
        loggers=MosaicMLLogger(),
    )
    trainer.fit()

    if dist.get_global_rank() == 0:
        assert len(mock_mapi.run_metadata[run_name].keys()) > 0
    else:
        assert len(mock_mapi.run_metadata.keys()) == 0


def test_metric_partial_filtering(monkeypatch):
    mock_mapi = MockMAPI()
    monkeypatch.setattr(mcli, 'update_run_metadata', mock_mapi.update_run_metadata)
    run_name = 'small_chungus'
    monkeypatch.setenv('RUN_NAME', run_name)

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='1ep',
        loggers=MosaicMLLogger(ignore_keys=['loss', 'accuracy']),
    )
    trainer.fit()

    assert 'mosaicml/num_nodes' in mock_mapi.run_metadata[run_name]
    assert 'mosaicml/loss' not in mock_mapi.run_metadata[run_name]


def test_metric_full_filtering(monkeypatch):
    mock_mapi = MockMAPI()
    monkeypatch.setattr(mcli, 'update_run_metadata', mock_mapi.update_run_metadata)
    run_name = 'small_chungus'
    monkeypatch.setenv('RUN_NAME', run_name)

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='1ep',
        loggers=MosaicMLLogger(ignore_keys=['*']),
    )
    trainer.fit()

    assert len(mock_mapi.run_metadata[run_name].keys()) == 0


class SetWandBRunURL(Callback):
    """Sets run_url attribute on WandB for offline unit testing."""

    def __init__(self, run_url) -> None:
        self.run_url = run_url

    def init(self, state, event) -> None:
        for callback in state.callbacks:
            if isinstance(callback, WandBLogger):
                callback.run_url = self.run_url


def test_wandb_run_url(monkeypatch):
    mock_mapi = MockMAPI()
    monkeypatch.setattr(mcli, 'update_run_metadata', mock_mapi.update_run_metadata)
    run_name = 'small_chungus'
    monkeypatch.setenv('RUN_NAME', run_name)

    run_url = 'my_run_url'
    monkeypatch.setenv('WANDB_MODE', 'offline')

    Trainer(model=SimpleModel(), loggers=[
        MosaicMLLogger(),
        WandBLogger(),
    ], callbacks=[
        SetWandBRunURL(run_url),
    ])

    assert mock_mapi.run_metadata[run_name]['mosaicml/wandb/run_url'] == run_url
