# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory usage during training."""
import logging
from collections import deque
from typing import List, Optional

import torch

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

import os

import numpy as np
import pynvml
from slack_sdk.webhook import WebhookClient

from composer.core import Callback, State
from composer.core.time import Timestamp
from composer.loggers import Logger
from composer.utils import dist

log = logging.getLogger(__name__)

__all__ = ['HealthChecker']


class HealthChecker(Callback):
    """Checks for GPU health.

    This callback checks for GPU health by measuring GPU utilization across all
    involved GPUs/nodes and alerts if the range of utilizations exceeds
    a certain threshold.

    For example, if utilization across GPUs for this run are
    [30, 30, 45], then the range (45-30=15) would exceed a threshold of 10%.

    Only GPUs involved in this training run are checked.

    """

    def __init__(
        self,
        threshold=0.10,
        sample_freq=5,
        check_freq=120,
        wait=120,
        slack_webhook_url=None,
    ) -> None:
        self.threshold = threshold
        self.sample_freq = sample_freq
        self.check_freq = check_freq
        self.wait = wait
        self.slack_webhook_url = slack_webhook_url

        if not self.slack_webhook_url:
            self.slack_webhook_url = os.environ.get('SLACK_WEBHOOK_URL', None)

        self.last_sample = 0
        self.last_check = 0

        self.metrics = []
        if self._is_available():
            self.metrics.append(GPUUtilization())

    def init(self, state: State, logger: Logger) -> None:
        pass

    def after_train_batch(self, state: State, logger: Logger):
        if self._sample(state.timestamp):
            for metric in self.metrics:
                metric.sample()

        if self._check(state.timestamp):
            for metric in self.metrics:
                message, alert = metric.check()
                if alert and not metric.alerted:
                    self._alert(message)
                    metric.alerted = True
                metric.clear()

    def _sample(self, timestamp: Timestamp) -> bool:
        now = timestamp.total_wct.seconds

        if now < self.wait:
            return False

        if now - self.last_sample > self.sample_freq:
            self.last_sample = now
            return True

        return False

    def _check(self, timestamp: Timestamp) -> bool:
        now = timestamp.total_wct.seconds

        if now - self.last_check > self.check_freq:
            self.last_check = now
            return True
        return False

    def _alert(self, message: str) -> None:
        message = 'Found a potential issue!'

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

    alerted: bool = False

    def __init__(self, threshold=10) -> None:
        self.samples = deque()
        self.threshold = threshold

    def sample(self) -> None:
        if dist.get_local_rank == 0:
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)

    def _sample(self) -> Optional[List[float]]:
        # TODO: catch NVMLError
        samples = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        return samples

    def check(self, state: State) -> Tuple[str, bool]:
        if dist.get_local_rank == 0:
            average_sample = np.nanmean(self.samples, axis=0)
            if np.nanmax(average_sample) - np.nanmin(average_sample) > self.threshold:
                message = '{run_name} experiencing abnormal GPU utilizations on rank {rank}: {utils}'
                return message.format(
                    run_name=state.run_name,
                    rank=dist.node_rank,
                    utils=average_sample,
                ), True

        return None, False

    def clear(self) -> None:
        self.samples.clear()


class ECCErrors:

    alerted: bool = False

    def __init__(self, threshold=100) -> None:
        self.samples = deque()
        self.threshold = threshold

    def sample(self) -> None:
        if dist.get_local_rank == 0:
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)

    def _sample(self) -> Optional[float]:
        samples = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            samples.append(pynvml.nvmlDeviceGetMemoryErrorCounter(handle, 0, 0, 2))
        return samples

    def check(self, state: State) -> Tuple[str, bool]:
        if dist.get_local_rank == 0:
            min_counter = np.min(self.samples, axis=0)
            max_counter = np.max(self.samples, axis=0)
            gpus_with_error = np.where(max_counter - min_counter > self.threshold)
            if len(gpus_with_error) > 0:
                message = '{run_name} reporting high memory ECC error on rank {rank} for GPUs: {gpus}'
                ecc_data = ['GPU: {} ({} -> {})'.format(i, min_counter[i], max_counter[i]) for i in gpus_with_error]
                return message.format(run_name=state.run_name, rank=dist.node_rank, gpus=ecc_data), True

        return None, False

    def clear(self) -> None:
        self.samples.clear()
