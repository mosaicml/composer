# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from composer.core import Logger, State
from composer.core.logging.logger import LogLevel
from composer.loggers import mosaicml_logger
from composer.loggers.mosaicml_logger import MosaicMLLoggerBackend
import pytest
from _pytest.monkeypatch import MonkeyPatch

from composer.loggers import MosaicMLLoggerBackendHparams


@pytest.mark.timeout(30)
def test_mosaic_logger(tmpdir: pathlib.Path, dummy_state: State, dummy_logger: Logger, monkeypatch: MonkeyPatch):
    creds_file = os.path.join(tmpdir, "api_creds.txt")
    with open(creds_file, 'w') as f:
        f.write("test_api_key")

    hparams = MosaicMLLoggerBackendHparams(job_id="job_id",
                                         sweep_id="sweep_id",
                                         creds_file=creds_file,
                                         flush_every_n_batches=1,
                                         max_logs_in_buffer=3)
    logger = hparams.initialize_object()

    data_logged = []

    def _mock_send_data(job_id, sweep_id, data):
        # data = self.queue.get()
        data_logged.append(data)
        # self.queue.task_done()

    # print(mosaicml_logger.__dict__)
    monkeypatch.setitem(mosaicml_logger.__dict__, "_send_data", _mock_send_data)
    # mosaicml_logger.
    # logger._send_data = _mock_send_data
    # monkeypatch.setattr(logger, "_send_data", _mock_send_data)

    for i in range(10):
        logger._log_metric(epoch=1, step=i, log_level=LogLevel.BATCH, data={f'data-{i}': 'value'})

    print('data logged', data_logged)

    logger.training_end(state=dummy_state, logger=dummy_logger)
    print('data logged', data_logged)
