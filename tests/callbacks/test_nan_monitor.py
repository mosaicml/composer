# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import NaNMonitor
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.mark.parametrize('should_nan', [True, False])
def test_nan_monitor(should_nan):
    # Make the callback
    nan_monitor = NaNMonitor()
    # Test model
    model = SimpleModel()
    # Construct the trainer and train. Make the LR huge to force a NaN, small if it shouldn't
    lr = 1e20 if should_nan else 1e-10
    trainer = Trainer(
        model=model,
        callbacks=nan_monitor,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters(), lr=lr),
        max_duration='100ba',
    )
    # If it should NaN, expect a RuntimeError
    if should_nan:
        with pytest.raises(RuntimeError) as excinfo:
            trainer.fit()
        assert 'Train loss contains a NaN.' in str(excinfo.value)
    else:
        trainer.fit()
