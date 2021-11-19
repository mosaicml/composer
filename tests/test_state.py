# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
import random
from dataclasses import Field, fields

import torch
import torch.nn.functional as F
from torch.functional import Tensor

from composer.algorithms.dummy import DummyHparams
from composer.core import State, types
from composer.core.state import DIRECT_SERIALIZATION_FIELDS, SKIP_SERIALIZATION_FIELDS, STATE_DICT_SERIALIZATION_FIELDS
from composer.datasets.dataloader import DataloaderHparams
from composer.models.base import BaseMosaicModel
from composer.utils import ensure_tuple
from tests.fixtures.models import SimpleBatchPairModel
from tests.utils.dataloader import get_dataloader


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


def get_dummy_state(model: BaseMosaicModel, train_dataloader: types.DataLoader, val_dataloader: types.DataLoader):
    optimizers = torch.optim.Adadelta(model.parameters())

    return State(model=model,
                 train_batch_size=random.randint(0, 100),
                 eval_batch_size=random.randint(0, 100),
                 grad_accum=random.randint(0, 100),
                 precision=types.Precision.AMP,
                 max_epochs=random.randint(0, 100),
                 epoch=random.randint(0, 100),
                 step=random.randint(0, 100),
                 loss=random_tensor(),
                 batch=(random_tensor(), random_tensor()),
                 outputs=random_tensor(),
                 train_dataloader=train_dataloader,
                 eval_dataloader=val_dataloader,
                 optimizers=optimizers,
                 schedulers=torch.optim.lr_scheduler.StepLR(optimizers, step_size=3),
                 algorithms=[DummyHparams().initialize_object()])


def is_field_serialized(f: Field) -> bool:
    if f.name in STATE_DICT_SERIALIZATION_FIELDS or f.name in DIRECT_SERIALIZATION_FIELDS:
        return True
    elif f.name in SKIP_SERIALIZATION_FIELDS:
        return False
    else:
        raise RuntimeError(f"Serialization method for field {f.name} not specified")


def assert_state_equivalent(state1: State, state2: State):
    # tested separately
    IGNORE_FIELDS = [
        'optimizers',
        'schedulers',
        'train_dataloader',
        'eval_dataloader',
        'algorithms',
        'callbacks',
        'precision_context',
    ]

    for f in fields(state1):
        if f.name in IGNORE_FIELDS or not is_field_serialized(f):
            continue

        var1 = getattr(state1, f.name)
        var2 = getattr(state2, f.name)

        if f.name == "model":
            for p, q in zip(state1.model.parameters(), state2.model.parameters()):
                torch.testing.assert_allclose(p, q, atol=1e-2, rtol=1e-2)
        elif isinstance(var1, types.Tensor):
            assert (var1 == var2).all()
        else:
            assert var1 == var2


def train_one_step(state: State, batch: types.Batch) -> None:
    x, y = batch
    state.batch = batch

    state.outputs = state.model(state.batch_pair)
    assert isinstance(y, Tensor)

    state.loss = F.cross_entropy(state.outputs, y)
    state.loss.backward()
    assert state.optimizers is not None
    assert state.schedulers is not None
    for optimizer in ensure_tuple(state.optimizers):
        optimizer.step()
    for scheduler in ensure_tuple(state.schedulers):
        scheduler.step()
    state.step += 1


def get_batch(model: SimpleBatchPairModel, dataloader_hparams: DataloaderHparams):
    dataset_hparams = model.get_dataset_hparams(1, drop_last=False, shuffle=False)
    dataloader_spec = dataset_hparams.initialize_object()
    dataloader = get_dataloader(dataloader_spec, dataloader_hparams, 1)
    for batch in dataloader:
        return batch
    raise RuntimeError("No batch in dataloader")


def test_state_serialize(tmpdir: pathlib.Path, dummy_model: BaseMosaicModel,
                         dummy_dataloader_hparams: DataloaderHparams, dummy_train_dataloader: types.DataLoader,
                         dummy_val_dataloader: types.DataLoader):

    assert isinstance(dummy_model, SimpleBatchPairModel)

    state1 = get_dummy_state(dummy_model, dummy_train_dataloader, dummy_val_dataloader)
    state2 = get_dummy_state(dummy_model, dummy_train_dataloader, dummy_val_dataloader)

    # train one step to set the optimizer states
    batch = get_batch(dummy_model, dummy_dataloader_hparams)
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
    batch = get_batch(dummy_model, dummy_dataloader_hparams)
    train_one_step(state1, batch)
    train_one_step(state2, batch)

    # both states should have equivalent state, model parameters, loss, and outputs
    assert_state_equivalent(state1, state2)
