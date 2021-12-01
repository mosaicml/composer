# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import os
import sys
import uuid
import warnings
from queue import Queue
from threading import Thread
from typing import Optional

import requests

from composer.core.logging import LogLevel, RankZeroLoggerBackend, TLogData
from composer.core.types import JSON, Logger, State, StateDict

_MOSAICML_API_KEY_ENV = "MOSAICML_LOGGER_API_KEY"
_MOSAICML_LOGGER_URL = "https://api.mosaicml.com/v0/log/metric"
_MOSAICML_UPSERT_RUN_URL = "https://api.mosaicml.com/v0/log/run"

_RUN_STATUS = "RUNNING"
_STOP_LOGGING_SIGNAL = "STOP_LOGGING"

log = logging.getLogger(__name__)


def _send_data(run_id: str, experiment_name: str, data: JSON):
    try:
        response = requests.post(_MOSAICML_LOGGER_URL,
                                 headers={"X-MosaicML-API-key": os.environ.get(_MOSAICML_API_KEY_ENV, "")},
                                 json={
                                     "runID": run_id,
                                     "experimentName": experiment_name,
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


def _upsert_run(run_id: str, run_name: str, experiment_name: str, run_config: Optional[JSON]):
    run_config = run_config if run_config is not None else {}
    # For now, terminate the process if the request does not succeed because we definitely want the run
    # to exist in the backend before logging
    # TODO: Better error handling
    try:
        response = requests.post(_MOSAICML_UPSERT_RUN_URL,
                                 headers={"X-MosaicML-API-key": os.environ.get(_MOSAICML_API_KEY_ENV, "")},
                                 json={
                                     "runID": run_id,
                                     "runName": run_name,
                                     "experimentName": experiment_name,
                                     "run_config": run_config,
                                     "status": _RUN_STATUS,
                                 },
                                 timeout=5)
        if response.status_code >= 400:
            # Terminate the process in the case of a non-transient error
            log.error("Posting run upsert to MosaicML backend failed with response code "
                      f"{response.status_code} and message {response.text}.")
            sys.exit(1)
    except requests.exceptions.Timeout as e:
        # Notify the user in the case of a timeout and terminate the process
        log.error(f"MosaicML logger run upsert timed out with error {e}. Terminating the process.")
        sys.exit(1)


class MosaicMLLoggerBackend(RankZeroLoggerBackend):
    """Log to the MosaicML backend.

    Args:
        run_name (str): The name of the run.
        experiment_name (str, optional): The name of the experiment to associate this run with. If not provided,
            a random name will be generated.
        run_id (str, optional): The id of the run to write logs for. If not provided, a random id will be generated.
            Note that not providing a run_id will result in a new run being created while providing a run_id
            will result in an existing run being updated.
        creds_file (str, optional): A file containing the MosaicML api_key. If not provided
            will default to the environment variable MOSAIC_API_KEY.
        flush_every_n_batches (int): Flush the log data buffer every n batches. (default: ``100``)
        max_logs_in_buffer (int): The maximum number of log entries allowed in the buffer before a
            forced flush. (default: ``1000``)
        run_config (JSON, optional): Additional configuration related to the run that will be stored along with
            the logs. For example, hyperparameters related to the training loop.
    """

    def __init__(self,
                 run_name: str,
                 experiment_name: Optional[str],
                 run_id: Optional[str] = None,
                 creds_file: Optional[str] = None,
                 flush_every_n_batches: int = 100,
                 max_logs_in_buffer: int = 1000,
                 run_config: Optional[JSON] = None) -> None:

        super().__init__()
        self.skip_logging = False
        self.run_name = run_name
        self.run_id = run_id  # if None will be set in training_start
        if experiment_name is None:
            experiment_name = f"experiment_{str(uuid.uuid4())}"
            log.info(f"experiment_name was None, set experiment_name to random value {experiment_name}")
        self.experiment_name = experiment_name
        self.run_config = run_config

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

    def training_start(self, state: State, logger: Logger):
        del state, logger  # unused

        if self.skip_logging:
            return

        if self.run_id is None:
            self.run_id = str(uuid.uuid4())
            log.info(f"run_id was None, set run_id to random value {self.run_id}")
        # This has to happen in training_start as opposed to init in order to make
        # resume from checkpoint work (i.e. the run_id from the checkpoint is loaded
        # after Event.INIT is called)
        # https://github.com/mosaicml/composer/issues/43 will fix this
        _upsert_run(run_id=self.run_id,
                    run_name=self.run_name,
                    experiment_name=self.experiment_name,
                    run_config=self.run_config)

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
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_name": self.experiment_name,
            "buffered_data": self.buffered_data
        }

    def load_state_dict(self, state: StateDict) -> None:
        self.run_id = state["run_id"]
        self.run_name = state["run_name"]
        self.experiment_name = state["experiment_name"]
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

            _send_data(run_id=self.run_id, experiment_name=self.experiment_name, data=data)  # type: ignore

            self.queue.task_done()
