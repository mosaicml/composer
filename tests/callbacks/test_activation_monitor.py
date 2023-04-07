# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import scipy
import torch
from torch.utils.data import DataLoader

from composer.callbacks import ActivationMonitor
from composer.callbacks.activation_monitor import compute_kurtosis
from composer.loggers import InMemoryLogger
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from tests.common.datasets import RandomClassificationDataset
from tests.common.models import SimpleModel


def test_activation_monitor():

    with pytest.warns(Warning):
        activation_monitor = ActivationMonitor(interval='1ba')
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel()
    trainer = Trainer(
        model=model,
        callbacks=activation_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters()),
        max_duration='3ba',
    )
    trainer.fit()

    num_max_calls = len(in_memory_logger.data['activations/max/_output.0'])
    num_average_calls = len(in_memory_logger.data['activations/average/_output.0'])

    assert num_max_calls == num_average_calls
    assert num_max_calls == 3

    # Checking to make sure existing keys are in the memory logger dict
    assert 'activations/l2_norm/module.0_input.0' in in_memory_logger.data.keys()
    assert 'activations/max/module.0_input.0' in in_memory_logger.data.keys()
    assert 'activations/average/module.0_input.0' in in_memory_logger.data.keys()
    assert 'activations/kurtosis/module.0_input.0' in in_memory_logger.data.keys()


def test_kurtosis_computation():
    x = torch.randn((2, 3, 4))
    res = compute_kurtosis(x)
    scipy_res = scipy.stats.kurtosis(x.numpy(), axis=-1, fisher=False).mean()
    assert np.allclose(res.numpy(), scipy_res, rtol=1e-4)
