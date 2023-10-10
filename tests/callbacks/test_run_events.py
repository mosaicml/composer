# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

import mcli
from torch.utils.data import DataLoader

from composer.callbacks import RunEventsCallback
from composer.loggers import MosaicMLLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel
from tests.loggers.test_mosaicml_logger import MockMAPI


def is_datetime(date_str):
    try:
        datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        return True
    except ValueError:
        return False


def test_run_events(monkeypatch):
    mock_mapi = MockMAPI()
    monkeypatch.setattr(mcli, 'update_run_metadata', mock_mapi.update_run_metadata)
    run_name = 'test_run_name'
    monkeypatch.setenv('RUN_NAME', run_name)

    run_events_callback = RunEventsCallback()
    mosaic_logger = MosaicMLLogger()

    model = SimpleModel()
    trainer = Trainer(
        model=model,
        callbacks=run_events_callback,
        loggers=mosaic_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ep',
    )
    trainer.fit()

    assert 'mosaicml/model_initialized_dt' in mock_mapi.run_metadata[run_name]
    assert is_datetime(mock_mapi.run_metadata[run_name]['mosaicml/model_initialized_dt'])
