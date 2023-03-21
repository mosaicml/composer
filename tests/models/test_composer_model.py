# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle
from typing import Iterable

import pytest
import torch
from torch.utils.data import DataLoader

from composer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleConvModel, SimpleModel


@pytest.mark.parametrize('model', [SimpleConvModel, SimpleModel])
def test_composermodel_torchscriptable(model):
    torch.jit.script(model())


@pytest.fixture()
def dataloader():
    return DataLoader(RandomClassificationDataset())


def test_model_access_to_logger(dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=2)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dataloader)
    assert model.logger is trainer.logger


def test_model_deepcopy(dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=2)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dataloader)
    assert model.logger is not None
    copied_model = copy.deepcopy(trainer.state.model)
    assert copied_model.logger is model.logger
    assert model.num_classes == copied_model.num_classes


def test_model_copy(dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=2)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dataloader)
    assert model.logger is not None
    copied_model = copy.copy(trainer.state.model)
    assert copied_model.logger is model.logger
    assert model.num_classes == copied_model.num_classes


def test_model_pickle(dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=2)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dataloader)
    assert model.logger is not None
    pickled_model = pickle.dumps(trainer.state.model)
    restored_model = pickle.loads(pickled_model)
    # after pickling the model, the restored loggers should be None, since the logger cannot be serialized
    assert restored_model.logger is None
    assert model.num_classes == restored_model.num_classes
