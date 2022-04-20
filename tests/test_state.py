# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
import random
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from composer.algorithms import ChannelsLastHparams
from composer.core import DataSpec, Evaluator, Precision, State
from composer.core.types import Batch, DataLoader
from composer.datasets.dataloader import DataLoaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.models.base import ComposerModel
from tests.fixtures.models import SimpleBatchPairModel


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


def get_dummy_state(model: ComposerModel, train_dataloader: DataLoader, val_dataloader: DataLoader):
    optimizers = torch.optim.Adadelta(model.parameters())

    evaluators = [Evaluator(label="dummy_label", dataloader=val_dataloader, metrics=model.metrics(train=False))]
    state = State(model=model,
                  grad_accum=random.randint(0, 100),
                  rank_zero_seed=random.randint(0, 100),
                  precision=Precision.AMP,
                  max_duration=f"{random.randint(0, 100)}ep",
                  train_dataloader=train_dataloader,
                  evaluators=evaluators,
                  optimizers=optimizers,
                  algorithms=[ChannelsLastHparams().initialize_object()])
    state.schedulers = torch.optim.lr_scheduler.StepLR(optimizers, step_size=3)
    state.loss = random_tensor()
    state.batch = (random_tensor(), random_tensor())
    state.outputs = random_tensor()
    return state


def _check_item(item1: Any, item2: Any, path: str):
    if item1 is None:
        assert item2 is None, f"{path} differs: {item1} != {item2}"
        return
    if isinstance(item1, (str, float, int, bool)):
        assert type(item1) == type(item2)
        assert item1 == item2, f"{path} differs: {item1} != {item2}"
        return
    if isinstance(item1, torch.Tensor):
        assert isinstance(item2, torch.Tensor)
        # Using a high tolerance, as deepspeed non-determinisim can cause
        # metric values to be off.
        assert item1.allclose(item2, rtol=0.1, atol=0.1), f"{path} differs"
        return
    if isinstance(item1, dict):
        assert isinstance(item2, dict), f"{path} differs: {item1} != {item2}"
        _check_dict_recursively(item1, item2, path)
        return
    if isinstance(item1, list):
        assert isinstance(item2, list), f"{path} differs: {item1} != {item2}"
        _check_list_recursively(item1, item2, path)
        return
    raise NotImplementedError(f"Unsupported item type: {type(item1)}")


def _check_list_recursively(list1: List[Any], list2: List[Any], path: str):
    assert len(list1) == len(list2), f"{path} differs: {list1} != {list2}"
    for i, (item1, item2) in enumerate(zip(list1, list2)):
        _check_item(item1, item2, f"{path}/{i}")


def _check_dict_recursively(dict1: Dict[str, Any], dict2: Dict[str, Any], path: str):
    assert len(dict1) == len(dict2), f"{path} differs: {dict1} != {dict2}"
    for k, val1 in dict1.items():
        val2 = dict2[k]
        _check_item(val1, val2, f"{path}/{k}")


def assert_state_equivalent(state1: State, state2: State):
    assert state1.serialized_attributes == state2.serialized_attributes
    assert state1.is_model_deepspeed == state2.is_model_deepspeed

    _check_dict_recursively(state1.state_dict(), state2.state_dict(), "")


def train_one_step(state: State, batch: Batch) -> None:
    _, y = batch
    state.batch = batch

    state.outputs = state.model(state.batch_pair)
    assert isinstance(y, Tensor)

    state.loss = F.cross_entropy(state.outputs, y)
    state.loss.backward()
    for optimizer in state.optimizers:
        optimizer.step()
    for scheduler in state.schedulers:
        scheduler.step()
    state.timer.on_batch_complete(len(batch))


def get_batch(dataset_hparams: DatasetHparams, dataloader_hparams: DataLoaderHparams) -> Batch:
    dataloader = dataset_hparams.initialize_object(batch_size=2, dataloader_hparams=dataloader_hparams)
    if isinstance(dataloader, DataSpec):
        dataloader = dataloader.dataloader
    for batch in dataloader:
        return batch
    raise RuntimeError("No batch in dataloader")


def test_state_serialize(tmpdir: pathlib.Path, dummy_model: ComposerModel, dummy_dataloader_hparams: DataLoaderHparams,
                         dummy_train_dataset_hparams: DatasetHparams, dummy_train_dataloader: DataLoader,
                         dummy_val_dataset_hparams: DatasetHparams, dummy_val_dataloader: DataLoader):

    assert isinstance(dummy_model, SimpleBatchPairModel)

    state1 = get_dummy_state(dummy_model, dummy_train_dataloader, dummy_val_dataloader)
    state2 = get_dummy_state(dummy_model, dummy_train_dataloader, dummy_val_dataloader)

    # train one step to set the optimizer states
    batch = get_batch(dummy_train_dataset_hparams, dummy_dataloader_hparams)
    train_one_step(state1, batch)

    # load from state1 to state2
    state_dict = state1.state_dict()
    filepath = str(tmpdir / "state.pt")
    torch.save(state_dict, filepath)
    state_dict_2 = torch.load(filepath, map_location="cpu")
    state2.load_state_dict(state_dict_2)
    # Make sure there was nothing wrong serialization/deserialization of permanent
    assert_state_equivalent(state1, state2)

    # train both for one step on another sample
    batch = get_batch(dummy_train_dataset_hparams, dummy_dataloader_hparams)
    train_one_step(state1, batch)
    train_one_step(state2, batch)

    # both states should have equivalent state, model parameters, loss, and outputs
    assert_state_equivalent(state1, state2)
