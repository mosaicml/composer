# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import NaNChecker
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


@pytest.mark.parametrize('should_nan', [True, False])
def test_nan_checker(should_nan):
    # Make the callback
    nan_checker = NaNChecker()
    # Test model
    model = SimpleModel()
    # Construct the trainer and train. Make the LR huge to force a NaN, small if it shouldn't
    lr = 1e10 if should_nan else 1e-10
    trainer = Trainer(
        model=model,
        callbacks=nan_checker,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters(), lr=lr),
        max_duration='10ba',
    )
    # If it should NaN, expect a RuntimeError
    if should_nan:
        with pytest.raises(RuntimeError) as excinfo:
            trainer.fit()
        assert 'Train loss contains a NaN.' in str(excinfo.value)
    else:
        trainer.fit()
