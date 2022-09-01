# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Tuple

import pytest
import torch
import torch.utils.data
from torch.optim import Optimizer

from composer.core import Precision, State
from composer.core.types import PyTorchScheduler
from tests.common import SimpleModel


@pytest.fixture
def dummy_in_shape() -> Tuple[int, ...]:
    return 1, 5, 5


@pytest.fixture
def dummy_num_classes() -> int:
    return 2


@pytest.fixture()
def dummy_train_batch_size() -> int:
    return 16


@pytest.fixture()
def dummy_val_batch_size() -> int:
    return 32


@pytest.fixture()
def dummy_train_n_samples() -> int:
    return 1000


@pytest.fixture
def dummy_model(dummy_in_shape: Tuple[int, ...], dummy_num_classes: int) -> SimpleModel:
    return SimpleModel(num_features=dummy_in_shape[0], num_classes=dummy_num_classes)


@pytest.fixture
def dummy_optimizer(dummy_model: SimpleModel):
    return torch.optim.SGD(dummy_model.parameters(), lr=0.001)


@pytest.fixture
def dummy_scheduler(dummy_optimizer: Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(dummy_optimizer, lambda _: 1.0)


@pytest.fixture()
def dummy_state(
    dummy_model: SimpleModel,
    dummy_train_dataloader: Iterable,
    dummy_optimizer: Optimizer,
    dummy_scheduler: PyTorchScheduler,
    rank_zero_seed: int,
    request: pytest.FixtureRequest,
) -> State:
    if request.node.get_closest_marker('gpu') is not None:
        # If using `dummy_state`, then not using the trainer, so move the model to the correct device
        dummy_model = dummy_model.cuda()
    state = State(
        model=dummy_model,
        run_name='dummy_run_name',
        precision=Precision.FP32,
        grad_accum=1,
        rank_zero_seed=rank_zero_seed,
        optimizers=dummy_optimizer,
        max_duration='10ep',
    )
    state.schedulers = dummy_scheduler
    state.set_dataloader(dummy_train_dataloader, 'train')

    return state
