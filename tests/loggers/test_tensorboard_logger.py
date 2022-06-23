# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from composer.loggers import TensorboardLogger
from tests.fixtures import dummy_fixtures

# To satisfy pyright.
dummy_state = dummy_fixtures.dummy_state
composer_trainer_hparams = dummy_fixtures.composer_trainer_hparams


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
