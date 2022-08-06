# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Type

import pytest
import torch
from torch.utils.data import DataLoader

from composer.core.callback import Callback
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.callbacks.callback_settings import get_cb_kwargs, get_cbs_and_marks
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('callback_cls', get_cbs_and_marks(callbacks=True))
def test_logged_data_is_json_serializable(callback_cls: Type[Callback]):
    """Test that all logged data is json serializable, which is a requirement to use wandb."""
    pytest.importorskip('wandb', reason='wandb is optional')
    from wandb.sdk.data_types.base_types.wb_value import WBValue
    callback_kwargs = get_cb_kwargs(callback_cls)
    callback = callback_cls(**callback_kwargs)
    logger = InMemoryLogger()  # using an in memory logger to manually validate json serializability
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

    for log_calls in logger.data.values():
        for timestamp, data in log_calls:
            del timestamp  # unused
            # manually filter out custom W&B data types and tensors, which are allowed, but cannot be json serialized
            if isinstance(data, (WBValue, torch.Tensor)):
                continue
            json.dumps(data)
