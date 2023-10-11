# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import DataLoader

from composer.callbacks import RunEventsCallback
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


def test_run_events():
    run_events_callback = RunEventsCallback()
    logger = InMemoryLogger()

    model = SimpleModel()
    trainer = Trainer(
        model=model,
        callbacks=run_events_callback,
        loggers=logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        eval_dataloader=DataLoader(RandomClassificationDataset()),
        max_duration='1ep',
    )
    trainer.fit()

    assert 'model_initialized_dt' in logger.data.keys()
    assert isinstance(logger.data['model_initialized_dt'][0][1], float)
