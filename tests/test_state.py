# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import random
from unittest.mock import Mock

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from composer.algorithms import ChannelsLast
from composer.core import DataSpec, Precision, State
from composer.core import state as state_module
from composer.core.types import Batch
from composer.datasets.dataset_hparams import DataLoaderHparams, DatasetHparams
from tests.common import SimpleModel, assert_state_equivalent


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


def get_dummy_state():
    model = SimpleModel()
    optimizers = torch.optim.Adadelta(model.parameters())
    state = State(model=model,
                  run_name=f'{random.randint(0, 100)}',
                  grad_accum=random.randint(0, 100),
                  rank_zero_seed=random.randint(0, 100),
                  precision=Precision.AMP,
                  max_duration=f'{random.randint(0, 100)}ep',
                  optimizers=optimizers,
                  algorithms=[ChannelsLast()])
    state.schedulers = torch.optim.lr_scheduler.StepLR(optimizers, step_size=3)
    state.loss = random_tensor()
    state.batch = (random_tensor(), random_tensor())
    state.outputs = random_tensor()
    return state


def train_one_step(state: State, batch: Batch) -> None:
    _, y = batch
    state.batch = batch

    for optimizer in state.optimizers:
        optimizer.zero_grad()

    state.outputs = state.model(state.batch)
    assert isinstance(y, Tensor)

    state.loss = F.cross_entropy(state.outputs, y)
    state.loss.backward()
    for optimizer in state.optimizers:
        optimizer.step()
    for scheduler in state.schedulers:
        scheduler.step()
    state.timestamp = state.timestamp.to_next_batch(len(batch))


def get_batch(dataset_hparams: DatasetHparams, dataloader_hparams: DataLoaderHparams) -> Batch:
    dataloader = dataset_hparams.initialize_object(batch_size=2, dataloader_hparams=dataloader_hparams)
    if isinstance(dataloader, DataSpec):
        dataloader = dataloader.dataloader
    for batch in dataloader:
        return batch
    raise RuntimeError('No batch in dataloader')


def test_state_serialize(
    tmp_path: pathlib.Path,
    dummy_dataloader_hparams: DataLoaderHparams,
    dummy_train_dataset_hparams: DatasetHparams,
):
    state1 = get_dummy_state()
    state2 = get_dummy_state()

    # train one step to set the optimizer states
    batch = get_batch(dummy_train_dataset_hparams, dummy_dataloader_hparams)
    train_one_step(state1, batch)

    # load from state1 to state2
    state_dict = state1.state_dict()
    filepath = str(tmp_path / 'state.pt')
    torch.save(state_dict, filepath)
    state_dict_2 = torch.load(filepath, map_location='cpu')
    state2.load_state_dict(state_dict_2)
    # Make sure there was nothing wrong serialization/deserialization of permanent
    assert_state_equivalent(state1, state2)

    # train both for one step on another sample
    batch = get_batch(dummy_train_dataset_hparams, dummy_dataloader_hparams)
    train_one_step(state1, batch)
    train_one_step(state2, batch)

    # both states should have equivalent state, model parameters, loss, and outputs
    assert_state_equivalent(state1, state2)


def test_state_batch_get_item(monkeypatch):
    state = get_dummy_state()
    state.batch = [1, 2]
    mock_batch_get = Mock()
    mock_batch_get.return_value = 7
    monkeypatch.setattr(state_module, 'batch_get', mock_batch_get)
    assert state.batch_get_item(2) == 7
    mock_batch_get.assert_called_once_with(state.batch, 2)


def test_state_batch_set_item(monkeypatch):
    state = get_dummy_state()
    state.batch = [1, 2]
    mock_batch_set = Mock()
    mock_batch_set.return_value = [7, 10]
    monkeypatch.setattr(state_module, 'batch_set', mock_batch_set)
    state.batch_set_item(key=1, value=154)
    mock_batch_set.assert_called_once_with([1, 2], key=1, value=154)


def test_state_batch_get_item_callable(monkeypatch):
    state = get_dummy_state()
    state.batch = [1, 2]
    mock_batch_get = Mock()
    monkeypatch.setattr(state_module, 'batch_get', mock_batch_get)
    getter = lambda x: x**2
    state.batch_get_item(key=getter)
    mock_batch_get.assert_called_once_with(state.batch, getter)
    state.batch_get_item(key=(getter, getter))
    mock_batch_get.assert_called_with(state.batch, (getter, getter))


def test_state_batch_set_item_callable(monkeypatch):
    state = get_dummy_state()
    state.batch = [1, 2]
    mock_batch_set = Mock()
    mock_batch_set.return_value = [4, 5]
    monkeypatch.setattr(state_module, 'batch_set', mock_batch_set)

    def setter(x, v):
        x[0] = v
        return x

    state.batch_set_item(key=setter, value=3)
    mock_batch_set.assert_called_once_with([1, 2], key=setter, value=3)
    state.batch_set_item(key=(setter, setter), value=4)
    mock_batch_set.assert_called_with([4, 5], key=(setter, setter), value=4)
