# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from composer.core import Callback
from composer.loggers import ConsoleLogger, LoggerDestination, ProgressBarLogger, SlackLogger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.trainer import Trainer
from tests.callbacks.callback_settings import (
    get_cb_kwargs,
    get_cb_model_and_datasets,
    get_cb_patches,
    get_cbs_and_marks,
)


@pytest.mark.parametrize('logger_cls', get_cbs_and_marks(loggers=True))
@pytest.mark.parametrize('callback_cls', get_cbs_and_marks(callbacks=True))
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_loggers_on_callbacks(logger_cls: type[LoggerDestination], callback_cls: type[Callback]):
    if logger_cls in [ProgressBarLogger, ConsoleLogger, SlackLogger]:
        pytest.skip()
    logger_kwargs = get_cb_kwargs(logger_cls)
    if issubclass(logger_cls, RemoteUploaderDownloader):
        # Ensure that the remote directory does not conflict with any directory used by callbacks
        logger_kwargs['backend_kwargs']['provider_kwargs']['key'] = './remote'
        os.makedirs(logger_kwargs['backend_kwargs']['provider_kwargs']['key'], exist_ok=True)
    logger = logger_cls(**logger_kwargs)
    callback_kwargs = get_cb_kwargs(callback_cls)
    callback = callback_cls(**callback_kwargs)
    model, train_dataloader, _ = get_cb_model_and_datasets(callback)
    maybe_patch_context = get_cb_patches(callback_cls)

    with maybe_patch_context:
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            train_subset_num_batches=2,
            max_duration='1ep',
            callbacks=callback,
            loggers=logger,
        )
        trainer.fit()
