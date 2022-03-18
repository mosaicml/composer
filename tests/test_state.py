# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
import random

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


def assert_state_equivalent(state1: State, state2: State):

    # tested separately
    IGNORE_FIELDS = [
        '_optimizers',
        '_schedulers',
        '_algorithms',
        '_callbacks',
    ]

    for field_name in state1.__dict__.keys():
        if field_name in IGNORE_FIELDS or field_name.lstrip("_") not in state1.serialized_attributes:
            continue

        var1 = getattr(state1, field_name)
        var2 = getattr(state2, field_name)

        if field_name == "model":
            assert state1.is_model_deepspeed == state2.is_model_deepspeed
            for p, q in zip(state1.model.parameters(), state2.model.parameters()):
                torch.testing.assert_allclose(p, q, atol=1e-2, rtol=1e-2)
        elif isinstance(var1, torch.Tensor):
            assert (var1 == var2).all()
        else:
            assert var1 == var2


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
