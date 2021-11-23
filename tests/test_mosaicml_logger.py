# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib

import requests
from _pytest.monkeypatch import MonkeyPatch

from composer.core import Logger, State
from composer.core.logging.logger import LogLevel
from composer.core.types import JSON
from composer.loggers import MosaicMLLoggerBackendHparams, mosaicml_logger


def test_mosaic_logger(tmpdir: pathlib.Path, dummy_state: State, dummy_logger: Logger, monkeypatch: MonkeyPatch):
    creds_file = os.path.join(tmpdir, "api_creds.txt")
    with open(creds_file, 'w') as f:
        f.write("test_api_key")

    hparams = MosaicMLLoggerBackendHparams(job_id="job_id",
                                           sweep_id="sweep_id",
                                           creds_file=creds_file,
                                           flush_every_n_batches=4,
                                           max_logs_in_buffer=3)
    logger = hparams.initialize_object()

    data_logged = []

    def _mock_send_data(job_id: str, sweep_id: str, data: JSON):
        del job_id, sweep_id
        data_logged.extend(list(data))
        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setitem(mosaicml_logger.__dict__, "_send_data", _mock_send_data)

    logger._training_start(dummy_state, dummy_logger)

    num_times_to_log = 10
    expected_data = []
    for i in range(num_times_to_log):
        data_point = {f'data-{i}': 'value'}
        logger._log_metric(epoch=1, step=i, log_level=LogLevel.BATCH, data=data_point)
        expected_data.append({
            "step": i,
            "epoch": 1,
            "job_id": "job_id",
            "sweep_id": "sweep_id",
            "data": data_point,
        })

    logger.training_end(state=dummy_state, logger=dummy_logger)

    assert len(data_logged) == len(expected_data)
    # Loop over data for clarity in assertion errors
    for i in range(len(expected_data)):
        assert data_logged[i] == expected_data[i]
