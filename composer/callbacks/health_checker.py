# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log memory usage during training."""
import logging
from collections import deque
from typing import Optional

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


# modified from https://github.com/wandb/wandb/blob/main/wandb/sdk/internal/system/assets/gpu.py
def gpu_in_use_by_this_process(gpu_handle) -> bool:

    pid = os.getpid()

    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        # do not report any gpu metrics if the base process cant be found
        return False

    our_processes = base_process.children(recursive=True)
    our_processes.append(base_process)

    our_pids = {process.pid for process in our_processes}

    compute_pids = {
        process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)  # type: ignore
    }
    graphics_pids = {
        process.pid for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu_handle)  # type: ignore
    }

    pids_using_device = compute_pids | graphics_pids

    return len(pids_using_device & our_pids) > 0


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

        self.alerted = False
        self.last_sample = 0
        self.last_check = 0

        self.metrics = [GPUUtilization()]

    def init(self, state: State, logger: Logger) -> None:
        pass

    def after_train_batch(self, state: State, logger: Logger):
        if self.alerted:
            # only alert once
            return

        if self._sample(state.timestamp):
            for metric in self.metrics:
                metric.sample()

        if self._check(state.timestamp):
            for metric in self.metrics:
                if metric.check():
                    self._alert(state, metric)
                    self.alerted = True
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

    def _alert(self, state: State, metric) -> None:
        message = 'Found a potential issue!'

        logging.warning(message)
        if self.slack_webhook_url:
            client = WebhookClient(url=self.slack_webhook_url)
            client.send(text=message)


class GPUUtilization:

    def __init__(self, threshold=10) -> None:
        self.samples = deque()
        self.threshold = threshold

    def sample(self) -> None:
        sample = self._sample()
        if sample is not None:
            self.samples.append(sample)

    def _sample(self) -> Optional[float]:
        device_count = pynvml.nvmlDeviceGetCount()  # type: ignore
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore
            if gpu_in_use_by_this_process(handle):
                return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,  # type: ignore
        return None

    def check(self) -> bool:
        sample = [s for s in self.samples if s is not None]
        average_sample = np.mean(sample) if sample else None
        all_samples = dist.all_gather_object(average_sample)

        return np.nanmax(all_samples) - np.nanmin(all_samples) > self.threshold

    def clear(self) -> None:
        self.samples.clear()


class ECCErrors:

    def __init__(self, threshold=100) -> None:
        self.samples = deque()
        self.threshold = threshold

    def sample(self) -> None:
        sample = self._sample()
        if sample is not None:
            self.samples.append(sample)

    def _sample(self) -> Optional[float]:
        device_count = pynvml.nvmlDeviceGetCount()  # type: ignore
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore
            if gpu_in_use_by_this_process(handle):
                return pynvml.nvmlDeviceGetMemoryErrorCounter(handle, 0, 0, 2)  # type: ignore
        return None

    def check(self) -> bool:
        sample = [s for s in self.samples if s is not None]
        return np.nanmax(sample) - np.nanmin(sample) > self.threshold

    def clear(self) -> None:
        self.samples.clear()
