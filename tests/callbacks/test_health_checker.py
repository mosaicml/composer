# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader

from composer.callbacks import HealthChecker
from composer.callbacks.health_checker import GPUUtilization
from composer.loggers import InMemoryLogger
from composer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleModel, device

# @pytest.mark.gpu
# def test_health_checker():
#     # Construct the trainer
#     health_checker = HealthChecker(wait=0, sample_freq=1, check_freq=2)
#     in_memory_logger = InMemoryLogger()
#     trainer = Trainer(
#         model=SimpleModel(),
#         callbacks=health_checker,
#         loggers=in_memory_logger,
#         train_dataloader=DataLoader(RandomClassificationDataset()),
#         max_duration='10000ba',
#         device='gpu',
#     )
#     trainer.fit()


@pytest.mark.gpu
def test_gpu_utilization():
    import pynvml
    HealthChecker._is_available()

    with patch.object(pynvml, 'nvmlDeviceGetUtilizationRates') as mock_method:
        mock_return = MagicMock()
        mock_return.gpu = MagicMock(side_effect=[100, 100, 80])
        mock_method.return_value = mock_return
        gpu_utilization = GPUUtilization()
        gpu_utilization.sample()
        gpu_utilization.sample()
        gpu_utilization.sample()
        gpu_utilization.check()
