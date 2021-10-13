# Copyright 2021 MosaicML. All Rights Reserved.

import pathlib
import random
from dataclasses import Field, fields

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from _pytest.monkeypatch import MonkeyPatch
from torch.functional import Tensor

from composer.algorithms.dummy import DummyHparams
from composer.core import State, types
from composer.core.state import DIRECT_SERIALIZATION_FIELDS, SKIP_SERIALIZATION_FIELDS, STATE_DICT_SERIALIZATION_FIELDS
from composer.models.resnet56_cifar10.resnet56_cifar10_hparams import CIFARResNetHparams
from composer.utils import ensure_tuple


def random_tensor(size=(4, 10)):
    return torch.rand(*size)


class CIFARResNet():
    """
    CIFAR ResNet model with corresponding random data generation
    """

    @staticmethod
    def get_model():
        return CIFARResNetHparams(num_classes=10).initialize_object()

    @staticmethod
    def random_data_pair(N=8, C=3, H=32, W=32, n_classes=10):
        input = torch.rand(N, C, H, W)
        label = torch.randint(0, n_classes, size=(N,))
        return input, label


def get_dummy_state():
    # TODO: use a smaller test model
    dummy_model = CIFARResNet.get_model()
    optimizers = torch.optim.Adadelta(dummy_model.parameters())

    return State(model=dummy_model,
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


def assert_state_equivalent(state1: State, state2: State, skip_transient_fields: bool):
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
        if f.name in IGNORE_FIELDS:
            continue
        if skip_transient_fields and not is_field_serialized(f):
            continue

        var1 = getattr(state1, f.name)
        var2 = getattr(state2, f.name)

        if f.name == "model":
            for p, q in zip(state1.model.parameters(), state2.model.parameters()):
                torch.testing.assert_allclose(p, q, atol=1e-2, rtol=1e-2)
        elif isinstance(var1, types.Tensor):
            torch.testing.assert_equal(var1, var2)
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


@pytest.mark.run_long
@pytest.mark.timeout(5)
def test_state_serialize(tmpdir: pathlib.Path):

    state1 = get_dummy_state()
    state2 = get_dummy_state()

    # train one step to set the optimizer states
    batch = CIFARResNet.random_data_pair()
    train_one_step(state1, batch)

    # load from state1 to state2
    state_dict = state1.state_dict()
    filepath = str(tmpdir / "state.pt")
    torch.save(state_dict, filepath)
    state_dict_2 = torch.load(filepath, map_location="cpu")
    state2.load_state_dict(state_dict_2)
    # Make sure there was nothing wrong serialization/deserialization of permanent
    assert_state_equivalent(state1, state2, skip_transient_fields=True)

    # train both for one step on another sample
    batch = CIFARResNet.random_data_pair()
    train_one_step(state1, batch)
    train_one_step(state2, batch)

    # both states should have equivalent state, model parameters, loss, and outputs
    assert_state_equivalent(state1, state2, skip_transient_fields=False)


def test_state_rank(monkeypatch: MonkeyPatch, dummy_state: State):
    monkeypatch.setattr(dist, "get_rank", lambda: 9)
    dummy_state.nproc_per_node = 4
    assert dummy_state.is_rank_set
    assert not dummy_state.is_rank_zero
    assert dummy_state.local_rank == 1
    assert dummy_state.global_rank == 9
