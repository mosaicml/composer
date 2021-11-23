# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import os
import sys
import warnings
from queue import Queue
from threading import Thread
from typing import Optional

import requests

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData
from composer.core.serializable import Serializable
from composer.core.types import JSON, Logger, State, StateDict

_MOSAICML_API_KEY_ENV = "MOSAICML_LOGGER_API_KEY"
_MOSAICML_LOGGER_URL = "https://api.mosaicml.com/v0/log/metric"
_JOB_ID_ENV = "MOSAICML_LOGGER_JOB_ID"

_STOP_LOGGING_SIGNAL = "STOP_LOGGING"

log = logging.getLogger(__name__)


def _send_data(job_id: str, data: JSON):
    try:
        response = requests.post(_MOSAICML_LOGGER_URL,
                                 headers={"X-MosaicML-API-key": os.environ.get(_MOSAICML_API_KEY_ENV, "")},
                                 json={
                                     "experimentID": job_id,
                                     "data": data,
                                 },
                                 timeout=5)
        if response.status_code >= 400:
            # Terminate the process in the case of a non-transient error
            log.error("Posting data to MosaicML backend failed with response code "
                      f"{response.status_code} and message {response.text}.")
            sys.exit(1)
    except requests.exceptions.Timeout as e:
        # Warn the user in the case of a timeout but don't retry for now
        log.warning(f"MosaicML logger timed out with error {e}. Logs were not sent to the backend.")


class MosaicMLLoggerBackend(RankZeroLoggerBackend):
    """Log to the MosaicML backend.

    Args:
        job_id (str, optional): The id of the job to write logs for.
        creds_file (str, optional): A file containing the MosaicML api_key. If not provided
            will default to the environment variable MOSAIC_API_KEY.
        flush_every_n_batches (int): Flush the log data buffer every n batches. (default: ``100``)
        max_logs_in_buffer (int): The maximum number of log entries allowed in the buffer before a
            forced flush. (default: ``1000``)
    """

    def __init__(self,
                 job_id: Optional[str] = None,
                 creds_file: Optional[str] = None,
                 flush_every_n_batches: int = 100,
                 max_logs_in_buffer: int = 1000) -> None:

        super().__init__()
        self.skip_logging = False
        self.job_id = job_id if job_id is not None else os.environ.get(_JOB_ID_ENV, None)
        if self.job_id is None:
            self.skip_logging = True
            warnings.warn("No job_id provided to MosaicLoggerBackend. MosaicML logger will be a no-op.")

        if creds_file:
            with open(creds_file, 'r') as f:
                os.environ[_MOSAICML_API_KEY_ENV] = str(f.read())

        if os.environ.get(_MOSAICML_API_KEY_ENV, None) is None:
            self.skip_logging = True
            warnings.warn(
                f"No api_key set for environment variable {_MOSAICML_API_KEY_ENV}. MosaicML logger will be a no-op.")

        self.buffered_data = []
        self.flush_every_n_batches = flush_every_n_batches
        self.max_logs_in_buffer = max_logs_in_buffer

        self.queue = Queue()
        self.thread = Thread(target=self._listen_to_queue)

    def _log_metric(self, epoch: int, step: int, log_level: LogLevel, data: TLogData):
        del log_level  # unused

        if self.skip_logging:
            return

        log_data = {
            "job_id": self.job_id,
            "epoch": epoch,
            "step": step,
            "data": data,
        }
        self.buffered_data.append(log_data)
        if len(self.buffered_data) > self.max_logs_in_buffer:
            self._flush_buffered_data()

    def init(self, state: State, logger: Logger) -> None:
        del state, logger  # unused

        if self.skip_logging:
            return

        log.info("Starting MosaicML logger thread.")

        # Start the logging thread
        self.thread.start()

    def batch_end(self, state: State, logger: Logger):
        del logger  # unused

        if self.skip_logging:
            return

        if (state.step + 1) % self.flush_every_n_batches == 0:
            self._flush_buffered_data()

    def training_end(self, state: State, logger: Logger):
        del state, logger  # unused

        if self.skip_logging:
            return

        # Flush any remaining logs on training end
        self._flush_buffered_data()

        self.queue.put_nowait(_STOP_LOGGING_SIGNAL)
        self.thread.join()

        log.info(f"MosaicML logger thread has exited.")

    def state_dict(self) -> StateDict:
        # Storing these fields in the state dict to support run resuming in the future
        return {"job_id": self.job_id, "buffered_data": self.buffered_data}

    def load_state_dict(self, state: StateDict) -> None:
        self.job_id = state["job_id"]
        self.buffered_data = state["buffered_data"]

    def _flush_buffered_data(self):
        if len(self.buffered_data) == 0:
            return

        data_to_write = self.buffered_data.copy()
        self.buffered_data = []

        self.queue.put_nowait(data_to_write)

    def _listen_to_queue(self):
        while True:
            data = self.queue.get(block=True)
            if data == _STOP_LOGGING_SIGNAL:
                log.info("MosaicML logger thread received stop logging signal.")
                self.queue.task_done()
                return

            _send_data(job_id=self.job_id, data=data)  # type: ignore

            self.queue.task_done()
