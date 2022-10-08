# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

from composer.core.state import State
from composer.loggers import Logger, LoggerDestination


def test_logger_file_upload(dummy_state: State):

    file_logged = False

    class DummyLoggerDestination(LoggerDestination):

        def upload_file(self, state: State, remote_file_name: str, file_path: pathlib.Path, *, overwrite: bool):
            nonlocal file_logged
            file_logged = True
            assert remote_file_name == 'foo'
            assert file_path.name == 'bar'
            assert overwrite

    logger = Logger(state=dummy_state, destinations=[DummyLoggerDestination()])
    logger.upload_file(
        remote_file_name='foo',
        file_path='bar',
        overwrite=True,
    )

    assert file_logged
