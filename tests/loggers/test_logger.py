# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

from composer.core.state import State
from composer.loggers import Logger, LoggerDestination, LogLevel


def test_logger_file_artifact(dummy_state: State):

    file_logged = False

    class DummyLoggerDestination(LoggerDestination):

        def log_file_artifact(self, state: State, log_level: LogLevel, artifact_name: str, file_path: pathlib.Path, *,
                              overwrite: bool):
            nonlocal file_logged
            file_logged = True
            assert artifact_name == 'foo'
            assert file_path.name == 'bar'
            assert overwrite

    logger = Logger(state=dummy_state, destinations=[DummyLoggerDestination()])
    logger.file_artifact(
        log_level='epoch',
        artifact_name='foo',
        file_path='bar',
        overwrite=True,
    )

    assert file_logged
