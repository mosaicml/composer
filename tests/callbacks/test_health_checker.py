# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import datetime
from unittest.mock import MagicMock, patch

import pytest

from composer import Timestamp
from composer.callbacks import HealthChecker
from composer.callbacks.health_checker import GPUUtilization
from composer.utils import dist
from tests.common import world_size

pynvml = pytest.importorskip('pynvml')
pytest.importorskip('slack_sdk')


class MockUtil:

    def __init__(self, util):
        self.gpu = util


@pytest.mark.gpu
@world_size(1, 2)
@pytest.mark.filterwarnings('ignore:.*HealthChecker is deprecated.*')
def test_gpu_utilization(world_size):
    assert HealthChecker._is_available()

    gpu_utilization_values = [
        MockUtil(100),
        MockUtil(10),
        MockUtil(100),
        MockUtil(100),
        MockUtil(100),
        MockUtil(100),
    ]

    with patch.multiple(pynvml,
                        nvmlDeviceGetUtilizationRates=MagicMock(side_effect=gpu_utilization_values),
                        nvmlDeviceGetCount=MagicMock(return_value=world_size)):

        gpu_utilization = GPUUtilization()
        gpu_utilization.sample()
        gpu_utilization.sample()
        gpu_utilization.sample()
        _, alert = gpu_utilization.check()

        should_alert = dist.get_local_rank() == 0 and world_size > 1
        assert alert == should_alert


@pytest.mark.gpu
@world_size(1, 2)
@pytest.mark.filterwarnings('ignore:.*HealthChecker is deprecated.*')
def test_health_checker(world_size):

    state = MagicMock()
    state.run_name = 'pytest-mock-run-kwei73'
    logger = MagicMock()

    health_checker = HealthChecker(
        sample_freq=1,
        window_size=3,
        wait=0,
    )

    gpu_utilization_values = [
        MockUtil(100),
        MockUtil(10),
        MockUtil(100),
        MockUtil(100),
        MockUtil(100),
        MockUtil(100),
    ]

    with patch.multiple(pynvml,
                        nvmlDeviceGetUtilizationRates=MagicMock(side_effect=gpu_utilization_values),
                        nvmlDeviceGetCount=MagicMock(return_value=world_size)):

        # collect data and checker
        for seconds in [1, 2, 3]:
            state.timestamp = Timestamp(total_wct=datetime.timedelta(seconds=seconds))
            health_checker.after_train_batch(state, logger)

        should_alert = dist.get_local_rank() == 0 and world_size > 1
        assert health_checker.metrics[0].alerted == should_alert


@pytest.mark.filterwarnings('ignore:.*HealthChecker is deprecated.*')
def test_health_checker_sampling():
    timestamp = Timestamp(total_wct=datetime.timedelta(seconds=0))

    health_checker = HealthChecker(
        sample_freq=1,
        window_size=5,
        wait=10,
    )

    config = [
        (5, False),  # before wait
        (11, True),
        (11.5, False),  # below sample frequency
        (12, True),
        (20, True),
        (11, False),  # no time travel
    ]

    for seconds, is_sample in config:
        timestamp = Timestamp(total_wct=datetime.timedelta(seconds=seconds))
        assert health_checker._sample(timestamp) == is_sample
