# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Check GPU Health during training."""
import logging
from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple

import torch

try:
    import pynvml
except ImportError:
    pynvml = None

import os

import numpy as np
from slack_sdk.webhook import WebhookClient

from composer.core import Callback, State
from composer.core.time import Timestamp
from composer.loggers import Logger
from composer.utils import dist

log = logging.getLogger(__name__)

__all__ = ['HealthChecker']


class HealthChecker(Callback):
    """Checks for GPU health.

    This callback checks for GPU health by tracking and alerting for abnormal
    GPU utilizations.

    For example, if the average utilization during the observation window is,
    [30, 30, 45], then the range (45-30=15) would exceed a threshold of 10%.

    Args:
        threshold (float, optional): Threshold of GPU utilization range to
            trigger an alert. Defaults to 10.
        sample_freq (int, optional): Sample frequency in seconds. Default: 5.
        window_size (int, optional): Window size in seconds. HealthChecker will
            check for abnormalities at this frequency. Default: 120.
        wait (int, optional): Seconds to wait for starting to sample. Default: 120.
        slack_webhook_url (str, optional): Slack URL to send alerts. Can also
            be set with the SLACK_WEBHOOK_URL environment variable. Default: None
        test_mode (bool, optional): If True, will send a test alert at the first check.
            Default: False
    """

    def __init__(
        self,
        threshold: float = 10,
        sample_freq: int = 5,
        window_size: int = 120,
        wait: int = 120,
        slack_webhook_url: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        self.sample_freq = sample_freq
        self.window_size = window_size
        self.wait = wait
        self.slack_webhook_url = slack_webhook_url
        self.test_mode = test_mode

        if not self.slack_webhook_url:
            self.slack_webhook_url = os.environ.get('SLACK_WEBHOOK_URL', None)

        self.last_sample = 0
        self.last_check = 0

        self.metrics = []
        if self._is_available():
            self.metrics.append(GPUUtilization(threshold))

    def init(self, state: State, logger: Logger) -> None:
        pass

    def after_train_batch(self, state: State, logger: Logger):
        if not self.metrics:
            return

        if self._sample(state.timestamp):
            for metric in self.metrics:
                metric.sample()

        if self._check(state.timestamp):
            for metric in self.metrics:
                message, alert = metric.check()
                if self.test_mode and message:
                    alert = True
                    message = '[**THIS IS A TEST**]' + message
                if alert and not metric.alerted:
                    self._alert(message, state)
                    metric.alerted = True
                metric.clear()

    def _sample(self, timestamp: Timestamp) -> bool:
        now = timestamp.total_wct.seconds

        if now < self.wait:
            return False

        if now - self.last_sample >= self.sample_freq:
            self.last_sample = now
            return True

        return False

    def _check(self, timestamp: Timestamp) -> bool:
        now = timestamp.total_wct.seconds

        if now - self.last_check >= self.window_size:
            self.last_check = now
            return True
        return False

    def _alert(self, message: str, state: State) -> None:
        prefix = '[{now}][{run_name}][node_rank={node_rank}]'.format(
            now=datetime.now(),
            run_name=state.run_name,
            node_rank=dist.get_node_rank(),
        )

        node_name = os.environ.get('NODENAME', None)
        if node_name is not None:
            prefix += f'[node={node_name}]'

        message = prefix + ' : ' + message

        logging.warning(message)
        if self.slack_webhook_url:
            client = WebhookClient(url=self.slack_webhook_url)
            client.send(text=message)

    @staticmethod
    def _is_available() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            pynvml.nvmlInit()  # type: ignore
            return True
        except pynvml.NVMLError_LibraryNotFound:  # type: ignore
            logging.warning('NVML not found, disabling GPU health checking')
        except ImportError:
            logging.warning('pynvml library not found, disabling GPU health checking.')
        except Exception as e:
            logging.warning(f'Error initializing NVML: {e}')

        return False


class GPUUtilization:
    """GPU Utilization Metric."""

    def __init__(self, threshold=10) -> None:
        self.samples = deque()
        self.threshold = threshold
        self.alerted = False

    def sample(self) -> None:
        if dist.get_local_rank() == 0:
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)

    def _sample(self) -> Optional[List]:
        try:
            samples = []
            device_count = pynvml.nvmlDeviceGetCount()  # type: ignore
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore
                samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)  # type: ignore
        except pynvml.NVMLError:  # type: ignore
            return None
        return samples

    def check(self) -> Tuple[Optional[str], bool]:
        if dist.get_local_rank() == 0:
            average_sample = np.nanmean(list(self.samples), axis=0)
            if np.nanmax(average_sample) - np.nanmin(average_sample) > self.threshold:
                message = f'Abnormal GPU utilizations: {average_sample}'
                return message, True
            else:
                message = f':+1: Normal GPU utilizations: {average_sample}'
                return message, False
        return None, False

    def clear(self) -> None:
        self.samples.clear()
