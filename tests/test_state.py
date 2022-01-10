# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
import random

import torch
import torch.nn.functional as F
from deepspeed.runtime.engine import DeepSpeedEngine
from torch.functional import Tensor

from composer.algorithms.dummy import DummyHparams
from composer.core import DataSpec, State, types
from composer.core.state import DIRECT_SERIALIZATION_FIELDS, SKIP_SERIALIZATION_FIELDS, STATE_DICT_SERIALIZATION_FIELDS
from composer.datasets.dataloader import DataloaderHparams
from composer.datasets.hparams import DatasetHparams
from composer.models.base import BaseMosaicModel
from tests.fixtures.models import SimpleBatchPairModel


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


def get_dummy_state(model: BaseMosaicModel, train_dataloader: types.DataLoader, val_dataloader: types.DataLoader):
    optimizers = torch.optim.Adadelta(model.parameters())

    evaluators = [types.Evaluator(label="dummy_label", dataset=val_dataloader, metrics=model.metrics(train=False))]
    state = State(model=model,
                  grad_accum=random.randint(0, 100),
                  precision=types.Precision.AMP,
                  max_epochs=random.randint(0, 100),
                  train_dataloader=train_dataloader,
                  evaluators=evaluators,
                  optimizers=optimizers,
                  schedulers=torch.optim.lr_scheduler.StepLR(optimizers, step_size=3),
                  algorithms=[DummyHparams().initialize_object()])
    state.epoch = random.randint(0, 100)
    state.step = random.randint(0, 100)
    state.loss = random_tensor()
    state.batch = (random_tensor(), random_tensor())
    state.outputs = random_tensor()
    return state


def is_field_serialized(field_name: str) -> bool:
    if field_name in STATE_DICT_SERIALIZATION_FIELDS or field_name in DIRECT_SERIALIZATION_FIELDS:
        return True
    elif field_name in SKIP_SERIALIZATION_FIELDS:
        return False
    else:
        raise RuntimeError(f"Serialization method for field {field_name} not specified")


def assert_state_equivalent(state1: State, state2: State):
    # tested separately
    IGNORE_FIELDS = [
        '_optimizers',
        '_schedulers',
        '_algorithms',
        '_callbacks',
    ]

    for field_name in state1.__dict__.keys():
        if field_name in IGNORE_FIELDS or not is_field_serialized(field_name):
            continue

        var1 = getattr(state1, field_name)
        var2 = getattr(state2, field_name)

        if field_name == "model":
            if isinstance(state1.model, DeepSpeedEngine):
                assert isinstance(state2.model, DeepSpeedEngine)
            for p, q in zip(state1.model.parameters(), state2.model.parameters()):
                torch.testing.assert_allclose(p, q, atol=1e-2, rtol=1e-2)
        elif isinstance(var1, types.Tensor):
            assert (var1 == var2).all()
        else:
            assert var1 == var2


def train_one_step(state: State, batch: types.Batch) -> None:
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
    state.step += 1


def get_batch(dataset_hparams: DatasetHparams, dataloader_hparams: DataloaderHparams) -> types.Batch:
    dataloader = dataset_hparams.initialize_object(batch_size=2, dataloader_hparams=dataloader_hparams)
    if isinstance(dataloader, DataSpec):
        dataloader = dataloader.dataloader
    for batch in dataloader:
        return batch
    raise RuntimeError("No batch in dataloader")


def test_state_serialize(tmpdir: pathlib.Path, dummy_model: BaseMosaicModel,
                         dummy_dataloader_hparams: DataloaderHparams, dummy_train_dataset_hparams: DatasetHparams,
                         dummy_train_dataloader: types.DataLoader, dummy_val_dataset_hparams: DatasetHparams,
                         dummy_val_dataloader: types.DataLoader):

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
