# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import random

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from composer.core import Precision, State
from composer.core.types import Batch
from tests.common import SimpleModel, assert_state_equivalent
from tests.common.datasets import RandomClassificationDataset


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


def get_dummy_state():
    model = SimpleModel()
    optimizers = torch.optim.Adadelta(model.parameters())
    state = State(
        model=model,
        run_name=f'{random.randint(0, 100)}',
        grad_accum=random.randint(0, 100),
        rank_zero_seed=random.randint(0, 100),
        precision=Precision.AMP,
        max_duration=f'{random.randint(0, 100)}ep',
        optimizers=optimizers,
    )
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
    assert isinstance(y, torch.Tensor)

    state.loss = F.cross_entropy(state.outputs, y)
    state.loss.backward()
    for optimizer in state.optimizers:
        optimizer.step()
    for scheduler in state.schedulers:
        scheduler.step()
    state.timestamp = state.timestamp.to_next_batch(len(batch))


def test_state_serialize(tmp_path: pathlib.Path,):
    state1 = get_dummy_state()
    state2 = get_dummy_state()

    dataloader = DataLoader(
        dataset=RandomClassificationDataset(),
        batch_size=2,
    )

    dataloader_iter = iter(dataloader)

    # train one step to set the optimizer states
    batch = next(dataloader_iter)
    train_one_step(state1, batch)

    # load from state1 to state2
    state_dict = state1.state_dict()

    filepath = str(tmp_path / 'state.pt')
    torch.save(state_dict, filepath)

    state_dict_2 = torch.load(filepath, map_location='cpu')
    state2.load_state_dict(state_dict_2)

    # serialization/deserialization should be exact
    assert_state_equivalent(state1, state2)

    # train both for one step on another sample
    batch = next(dataloader_iter)
    train_one_step(state1, batch)
    train_one_step(state2, batch)

    # both states should have equivalent
    # state, model parameters, loss, and outputs
    assert_state_equivalent(state1, state2)


# yapf: disable
@pytest.mark.parametrize('batch,key,val', [
    ([1234, 5678], 0, 1234),
    ([1234, 5678], 1, 5678),
    ({'a': 1, 'b': 2}, 'a', 1),
    ({'a': 1, 'b': 2}, 'b', 2),
    (({'a': 1, 'b': 7}, {'c': 5}), lambda x: x[1]['c'], 5),
])
# yapf: enable
def test_state_batch_get_item(batch, key, val):
    state = get_dummy_state()
    state.batch = batch

    assert state.batch_get_item(key) == val


# yapf: disable
@pytest.mark.parametrize('batch,key,val', [
    ([1234, 5678], 0, 1111),
    ([1234, 5678], 1, 1111),
    ({'a': 1, 'b': 2}, 'a', 9),
    ({'a': 1, 'b': 2}, 'b', 9),
])
# yapf: enable
def test_state_batch_set_item(batch, key, val):
    state = get_dummy_state()
    state.batch = batch

    state.batch_set_item(key=key, value=val)
    assert state.batch_get_item(key) == val
