# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle
from typing import Iterable

import pytest
import torch

from composer.trainer import Trainer
from tests.common.models import SimpleConvModel, SimpleModel


@pytest.mark.parametrize('model', [SimpleConvModel, SimpleModel])
def test_composermodel_torchscriptable(model):
    torch.jit.script(model())


def test_model_access_to_logger(dummy_train_dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dummy_train_dataloader)
    assert model.logger is trainer.logger


def test_model_deepcopy(dummy_train_dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dummy_train_dataloader)
    assert model.logger is not None
    copied_model = copy.deepcopy(trainer.state.model)
    assert copied_model.logger is model.logger
    assert model.num_classes == copied_model.num_classes


def test_model_copy(dummy_train_dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dummy_train_dataloader)
    assert model.logger is not None
    copied_model = copy.copy(trainer.state.model)
    assert copied_model.logger is model.logger
    assert model.num_classes == copied_model.num_classes


def test_model_pickle(dummy_train_dataloader: Iterable):
    model = SimpleModel(num_features=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration='1ep', train_dataloader=dummy_train_dataloader)
    assert model.logger is not None
    pickled_model = pickle.dumps(trainer.state.model)
    restored_model = pickle.loads(pickled_model)
    # after pickling the model, the restored loggers should be None, since the logger cannot be serialized
    assert restored_model.logger is None
    assert model.num_classes == restored_model.num_classes
