# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Type

import pytest
from torch.utils.data import DataLoader

from composer.core import Callback
from composer.loggers import LoggerDestination
from composer.loggers.object_store_logger import ObjectStoreLogger
from composer.trainer import Trainer
from tests.callbacks.callback_settings import get_cb_kwargs, get_cbs_and_marks
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('logger_cls', get_cbs_and_marks(loggers=True))
@pytest.mark.parametrize('callback_cls', get_cbs_and_marks(callbacks=True))
def test_loggers_on_callbacks(logger_cls: Type[LoggerDestination], callback_cls: Type[Callback]):
    logger_kwargs = get_cb_kwargs(logger_cls)
    if issubclass(logger_cls, ObjectStoreLogger):
        # Ensure that the remote directory does not conflict with any directory used by callbacks
        logger_kwargs['object_store_kwargs']['provider_kwargs']['key'] = './remote'
        os.makedirs(logger_kwargs['object_store_kwargs']['provider_kwargs']['key'], exist_ok=True)
    logger = logger_cls(**logger_kwargs)
    callback_kwargs = get_cb_kwargs(callback_cls)
    callback = callback_cls(**callback_kwargs)
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=DataLoader(RandomClassificationDataset()),
        train_subset_num_batches=2,
        max_duration='1ep',
        callbacks=callback,
        loggers=logger,
        compute_training_metrics=True,
    )
    trainer.fit()
