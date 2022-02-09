# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import os
import sys
import uuid
import warnings
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional

import requests

from composer.core.logging import LogLevel, TLogData
from composer.core.logging.base_backend import LoggerCallback
from composer.core.logging.logger import format_log_data_as_json
from composer.core.time import Timestamp
from composer.core.types import JSON, Logger, State, StateDict
from composer.utils import dist
from composer.utils.string_enum import StringEnum

_MOSAICML_API_KEY_ENV = "MOSAICML_LOGGER_API_KEY"
_MOSAICML_LOGGER_URL = "https://api.mosaicml.com/v0/log/metric"
_MOSAICML_UPSERT_RUN_URL = "https://api.mosaicml.com/v0/log/run"

_STOP_LOGGING_SIGNAL = "STOP_LOGGING"

log = logging.getLogger(__name__)


class RunStatus(StringEnum):
    RUNNING = "running"
    COMPLETED = "completed"


class RunType(StringEnum):
    BENCHMARKING = "benchmarking"
    TRAINING = "training"


def _send_data(run_id: str, experiment_id: str, data: List[TLogData]):
    try:
        response = requests.post(_MOSAICML_LOGGER_URL,
                                 headers={"X-MosaicML-API-key": os.environ.get(_MOSAICML_API_KEY_ENV, "")},
                                 json={
                                     "runID": run_id,
                                     "experimentID": experiment_id,
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


def _upsert_run(run_id: str,
                run_name: str,
                run_type: RunType,
                experiment_name: str,
                run_status: RunStatus,
                run_config: Optional[Dict[str, JSON]] = None):
    # This functionality will eventually be moved to server-side
    run_config = run_config if run_config is not None else {}
    # For now, terminate the process if the request does not succeed because we definitely want the run
    # to exist in the backend before logging
    try:
        response = requests.post(_MOSAICML_UPSERT_RUN_URL,
                                 headers={"X-MosaicML-API-key": os.environ.get(_MOSAICML_API_KEY_ENV, "")},
                                 json={
                                     "runID": run_id,
                                     "runName": run_name,
                                     "runType": run_type.value,
                                     "experimentName": experiment_name,
                                     "runConfig": run_config,
                                     "status": run_status.value,
                                 },
                                 timeout=5)
        if response.status_code >= 400:
            # Terminate the process in the case of a non-transient error
            log.error("Posting run upsert to MosaicML backend failed with response code "
                      f"{response.status_code} and message {response.text}.")
            sys.exit(1)
        response_json = response.json()
        return response_json["experimentID"] if "experimentID" in response_json else None
    except requests.exceptions.Timeout as e:
        # Notify the user in the case of a timeout and terminate the process
        log.error(f"MosaicML logger run upsert timed out with error {e}. Terminating the process.")
        sys.exit(1)


class MosaicMLLogger(LoggerCallback):
    """Log to the MosaicML backend.

    Args:
        run_name (str): The name of the run.
        run_type (RunType): The type of the run.
        experiment_name (str, optional): The name of the experiment to associate this run with. If not provided,
            a random name will be generated.
        run_id (str, optional): The id of the run to write logs for. If not provided, a random id will be generated.
            Note that not providing a run_id will result in a new run being created while providing a run_id
            will result in an existing run being updated.
        creds_file (str, optional): A file containing the MosaicML api_key. If not provided
            will default to the environment variable MOSAIC_API_KEY. A valid key must be present or this logger will be
            a no-op.
        flush_every_n_batches (int): Flush the log data buffer every n batches. (default: ``500``)
        max_logs_in_buffer (int): The maximum number of log entries allowed in the buffer before a
            forced flush. (default: ``1000``)
        config (Dict[str, `~composer.core.types.JSON`], optional): Additional configuration related to the
            run that will be stored along with the logs. For example, hyperparameters related to the training loop.
    """

    def __init__(self,
                 run_name: str,
                 run_type: RunType,
                 experiment_name: Optional[str],
                 run_id: Optional[str] = None,
                 creds_file: Optional[str] = None,
                 flush_every_n_batches: int = 500,
                 max_logs_in_buffer: int = 1000,
                 log_level: LogLevel = LogLevel.EPOCH,
                 config: Optional[Dict[str, JSON]] = None) -> None:

        super().__init__()
        self.skip_logging = dist.get_global_rank() != 0
        self.log_level = log_level
        self.run_name = run_name
        self.run_type = run_type
        self.run_id = run_id  # if None, will via load_state_dict or in _flush_buffered_data
        self.experiment_id = None  # will be set in _flush_buffered_data
        if experiment_name is None:
            experiment_name = f"experiment_{str(uuid.uuid4())}"
            log.info(f"experiment_name was None, set experiment_name to random value {experiment_name}")
        self.experiment_name = experiment_name
        self.run_config = config

        if creds_file:
            with open(creds_file, 'r') as f:
                os.environ[_MOSAICML_API_KEY_ENV] = str(f.read().strip())

        if os.environ.get(_MOSAICML_API_KEY_ENV, None) is None:
            self.skip_logging = True
            warnings.warn(
                f"No api_key set for environment variable {_MOSAICML_API_KEY_ENV}. MosaicML logger will be a no-op.")

        self.buffered_data = []
        self.flush_every_n_batches = flush_every_n_batches
        self.max_logs_in_buffer = max_logs_in_buffer

        self.queue = Queue()
        self.thread = Thread(target=self._listen_to_queue, daemon=True, name="mosaicml-logger-thread")

        self._training_started = False

    def will_log(self, state: State, log_level: LogLevel) -> bool:
        del state  # unused
        return log_level <= self.log_level

    def log_metric(self, timestamp: Timestamp, log_level: LogLevel, data: TLogData):
        del log_level  # unused

        if self.skip_logging:
            return

        formatted_data = format_log_data_as_json(data)
        log_data = {
            "epoch": int(timestamp.epoch),
            "step": int(timestamp.batch),
        }
        log_data.update(formatted_data)  # type: ignore
        self.buffered_data.append(log_data)
        if len(self.buffered_data) > self.max_logs_in_buffer:
            self._flush_buffered_data()

    def batch_end(self, state: State, logger: Logger):
        del logger  # unused

        self._training_started = True

        if self.skip_logging:
            return

        if int(state.timer.batch) % self.flush_every_n_batches == 0:
            self._flush_buffered_data()

    def post_close(self):
        # Write any relevant logs from other callback's close() functions here
        if self.skip_logging:
            return

        # Flush any remaining logs on training end
        self._flush_buffered_data()

        assert self.run_id is not None, "run ID is set in self._flush_buffered_data"

        log.info(f"Updating run status to {RunStatus.COMPLETED}")
        _upsert_run(run_id=self.run_id,
                    run_name=self.run_name,
                    run_type=self.run_type,
                    experiment_name=self.experiment_name,
                    run_status=RunStatus.COMPLETED,
                    run_config=self.run_config)

        self.queue.put_nowait(_STOP_LOGGING_SIGNAL)
        self.thread.join()

        log.info("MosaicML logger thread has exited.")

    def state_dict(self) -> StateDict:
        # Storing these fields in the state dict to support run resuming in the future
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_name": self.experiment_name,
            "buffered_data": self.buffered_data
        }

    def load_state_dict(self, state: StateDict) -> None:
        self.run_id = str(state["run_id"])
        self.run_name = str(state["run_name"])
        self.experiment_name = str(state["experiment_name"])
        self.buffered_data = list(state["buffered_data"])

    def _flush_buffered_data(self):
        if len(self.buffered_data) == 0:
            return

        if not self._training_started:
            # We need to know that training started to ensure that we loaded the state
            # from the checkpoint (if we're resuming from the checkpoint) so the
            # run ID is set correctly
            return

        if self.thread.ident is None:  # if the thread is not started
            # run_id could be None if not passed in via __init__ and not resuming from a checkpoint
            if self.run_id is None:
                self.run_id = str(uuid.uuid4())
                log.info(f"run_id was None, set run_id to random value {self.run_id}")

            self.experiment_id = _upsert_run(run_id=self.run_id,
                                             run_name=self.run_name,
                                             run_type=self.run_type,
                                             experiment_name=self.experiment_name,
                                             run_status=RunStatus.RUNNING,
                                             run_config=self.run_config)

            log.info("Starting MosaicML logger thread.")

            # Start the logging thread
            self.thread.start()

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

            assert self.run_id is not None, "run ID should be set before the thread is started"
            assert self.experiment_id is not None, "experiment ID should be set before the thread is started"

            _send_data(run_id=self.run_id, experiment_id=self.experiment_id, data=data)

            self.queue.task_done()
