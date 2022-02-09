# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from typing import Dict, List, Optional

from _pytest.monkeypatch import MonkeyPatch

from composer.core import Logger, State
from composer.core.logging.logger import LogLevel, TLogData
from composer.core.types import JSON
from composer.loggers import MosaicMLLoggerHparams, mosaicml_logger
from composer.loggers.mosaicml_logger import RunStatus, RunType


def test_mosaic_logger(tmpdir: pathlib.Path, dummy_state: State, dummy_logger: Logger, monkeypatch: MonkeyPatch):
    creds_file = os.path.join(tmpdir, "api_creds.txt")
    with open(creds_file, 'w') as f:
        f.write("test_api_key")

    flush_every_n_batches = 5
    max_logs_in_buffer = 3
    hparams = MosaicMLLoggerHparams(run_name="run_name",
                                    run_type=RunType.BENCHMARKING,
                                    experiment_name="experiment_name",
                                    run_id="run_id",
                                    creds_file=creds_file,
                                    flush_every_n_batches=flush_every_n_batches,
                                    max_logs_in_buffer=max_logs_in_buffer)
    logger = hparams.initialize_object()

    data_logged = []
    num_log_calls = 0

    def _mock_send_data(run_id: str, experiment_id: str, data: List[TLogData]):
        del run_id, experiment_id
        nonlocal num_log_calls
        num_log_calls += 1
        assert isinstance(data, list)
        data_logged.extend(data)

    def _mock_upsert_run(run_id: str,
                         run_name: str,
                         run_type: RunType,
                         experiment_name: str,
                         run_status: RunStatus,
                         run_config: Optional[Dict[str, JSON]] = None):
        del run_id, run_name, run_type, experiment_name, run_status, run_config  # unused
        return "experiment_id"

    # Replace the network call with a function that records logs sent
    monkeypatch.setitem(mosaicml_logger.__dict__, "_send_data", _mock_send_data)
    monkeypatch.setitem(mosaicml_logger.__dict__, "_upsert_run", _mock_upsert_run)

    # Start the logging thread
    logger.init(dummy_state, dummy_logger)

    # Call _log_metric and record expected results
    num_times_to_log = 10
    expected_data = []
    buffer_length = 0
    expected_log_calls = 0
    dummy_state.timer.on_epoch_complete()
    for i in range(num_times_to_log):
        data_point = {f'data-{i}': 'value'}
        logger.log_metric(timestamp=dummy_state.timer.get_timestamp(), log_level=LogLevel.BATCH, data=data_point)
        dummy_state.timer.on_batch_complete()
        logger.batch_end(dummy_state, dummy_logger)

        expected_log_point = {
            "step": i,
            "epoch": 1,
        }
        expected_log_point.update(data_point)  # type: ignore
        expected_data.append(expected_log_point)
        buffer_length += 1

        # Emulate buffer flushing
        if buffer_length > max_logs_in_buffer:
            expected_log_calls += 1
            buffer_length = 0
        if (i + 1) % flush_every_n_batches == 0 and buffer_length > 0:
            buffer_length = 0
            expected_log_calls += 1

    logger.post_close()

    assert num_log_calls == expected_log_calls
    assert len(data_logged) == len(expected_data)
    # Loop over data for clarity in assertion errors
    for i in range(len(expected_data)):
        assert data_logged[i] == expected_data[i]
