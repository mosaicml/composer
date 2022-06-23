# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from composer.loggers import TensorboardLogger
from composer.trainer import Trainer
from unittest.mock import Mock
from tests.fixtures import dummy_fixtures

# To satisfy pyright.
dummy_state = dummy_fixtures.dummy_state
composer_trainer_hparams = dummy_fixtures.composer_trainer_hparams


def test_tensorboard_logger_log_data(monkeypatch: pytest.MonkeyPatch, dummy_state):
    tbl = TensorboardLogger()
    mock_writer = Mock()
    add_scalar_fn = Mock()
    test_data = {'test': 3}
    state = dummy_state
    state.timestamp = Mock()
    state.timestamp.batch = 7
    mock_writer.add_scalar = add_scalar_fn
    monkeypatch.setattr(tbl, 'writer', mock_writer)
    tbl.log_data(state, log_level=Mock(), data=test_data)
    add_scalar_fn.assert_called_once_with('test', 3, global_step=7)


def test_tensorboard_logger_trainer(monkeypatch: pytest.MonkeyPatch, composer_trainer_hparams):
    tbl = TensorboardLogger()
    mock_log_data = Mock()
    monkeypatch.setattr(tbl, 'log_data', mock_log_data)
    trainer_hparams = composer_trainer_hparams
    trainer_hparams.loggers = [tbl]
    trainer_hparams.train_subset_num_batches = 1
    trainer_hparams.eval_subset_num_batches = 1
    trainer = trainer_hparams.initialize_object()
    trainer.fit()
    mock_log_data.assert_called()




