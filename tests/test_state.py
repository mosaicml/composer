# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
import random

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import composer
from composer.core import Batch, Precision, State
from composer.core.time import TimeUnit
from composer.devices import DeviceCPU, DeviceGPU
from composer.loggers import Logger
from tests.common import SimpleModel, assert_state_equivalent
from tests.common.datasets import RandomClassificationDataset


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


def get_dummy_state(request: pytest.FixtureRequest):
    model = SimpleModel()
    dataset = RandomClassificationDataset()
    dataloader = DataLoader(dataset, batch_size=4)
    optimizers = torch.optim.Adadelta(model.parameters())
    device = None
    for item in request.session.items:
        device = DeviceCPU() if item.get_closest_marker('gpu') is None else DeviceGPU()
        break
    assert device != None
    state = State(
        model=model,
        device=device,
        train_dataloader=dataloader,
        run_name=f'{random.randint(0, 100)}',
        rank_zero_seed=random.randint(0, 100),
        precision=Precision.AMP_FP16,
        max_duration=f'{random.randint(0, 100)}ep',
        optimizers=optimizers,
        device_train_microbatch_size=2,
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


def test_state_serialize(tmp_path: pathlib.Path, empty_logger: Logger, request: pytest.FixtureRequest):
    state1 = get_dummy_state(request)
    state2 = get_dummy_state(request)

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
    state2.load_state_dict(state_dict_2, empty_logger)

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
@pytest.mark.parametrize(
    'batch,key,val', [
        ([1234, 5678], 0, 1234),
        ([1234, 5678], 1, 5678),
        ({'a': 1, 'b': 2}, 'a', 1),
        ({'a': 1, 'b': 2}, 'b', 2),
        (({'a': 1, 'b': 7}, {'c': 5}), lambda x: x[1]['c'], 5),
    ],
)
# yapf: enable
def test_state_batch_get_item(batch, key, val, request: pytest.FixtureRequest):
    state = get_dummy_state(request)
    state.batch = batch

    assert state.batch_get_item(key) == val


# yapf: disable
@pytest.mark.parametrize(
    'batch,key,val', [
        ([1234, 5678], 0, 1111),
        ([1234, 5678], 1, 1111),
        ({'a': 1, 'b': 2}, 'a', 9),
        ({'a': 1, 'b': 2}, 'b', 9),
    ],
)
# yapf: enable
def test_state_batch_set_item(batch, key, val, request: pytest.FixtureRequest):
    state = get_dummy_state(request)
    state.batch = batch

    state.batch_set_item(key=key, value=val)
    assert state.batch_get_item(key) == val


@pytest.mark.parametrize('time_unit', [ # Does not test for TimeUnit.DURATION because max_duration cannot have TimeUnit.DURATION as its unit: https://github.com/mosaicml/composer/blob/1b9c6d3c0592183b947fd89890de0832366e33a7/composer/core/state.py#L628
    TimeUnit.EPOCH,
    TimeUnit.BATCH,
    TimeUnit.SAMPLE,
    TimeUnit.TOKEN,
])
def test_stop_training(time_unit: TimeUnit, request: pytest.FixtureRequest):
    state = get_dummy_state(request)
    state.max_duration = '10' + time_unit.value
    state.stop_training()
    if time_unit == TimeUnit.EPOCH:
        assert state.max_duration == '0' + TimeUnit.BATCH.value
    else:
        assert state.max_duration == '0' + time_unit.value


def test_composer_metadata_in_state_dict(tmp_path, request: pytest.FixtureRequest):
    state = get_dummy_state(request)
    save_path = pathlib.Path(tmp_path) / 'state_dict.pt'
    with open(save_path, 'wb') as _tmp_file:
        torch.save(state.state_dict(), _tmp_file)

    loaded_state_dict = torch.load(save_path)
    expected_env_info_keys = {
        'composer_version',
        'composer_commit_hash',
        'cpu_model',
        'cpu_count',
        'num_nodes',
        'gpu_model',
        'num_gpus_per_node',
        'num_gpus',
        'cuda_device_count',
    }
    actual_env_info_keys = set(loaded_state_dict['metadata']['composer_env_info'].keys())
    assert expected_env_info_keys == actual_env_info_keys
    assert loaded_state_dict['metadata']['composer_env_info']['composer_version'] == composer.__version__

    assert loaded_state_dict['metadata']['torch_version'] == torch.__version__
    assert loaded_state_dict['metadata']['device'] == 'cpu'
    assert loaded_state_dict['metadata']['precision'] == 'amp_fp16'
    assert loaded_state_dict['metadata']['world_size'] == 1
    assert loaded_state_dict['metadata']['device_train_microbatch_size'] == 2
    assert loaded_state_dict['metadata']['train_dataloader_batch_size'] == 4
