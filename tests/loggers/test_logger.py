# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib
from typing import List

import pytest

from composer.core.state import State
from composer.loggers import (ConsoleLogger, FileLogger, InMemoryLogger, Logger, LoggerDestination, MLFlowLogger,
                              MosaicMLLogger, ProgressBarLogger, RemoteUploaderDownloader, SlackLogger,
                              TensorboardLogger, WandBLogger)


@pytest.mark.parametrize('num_dest_instances_per_type', [0, 1, 2])
def test_logger_properties(dummy_state: State, num_dest_instances_per_type: int):
    logger_destination_classes = (ConsoleLogger, FileLogger, InMemoryLogger, MLFlowLogger, MosaicMLLogger,
                                  ProgressBarLogger, SlackLogger, TensorboardLogger, WandBLogger)
    destinations = []

    for _ in range(num_dest_instances_per_type):
        for logger_destination_type_cls in logger_destination_classes:
            logger_dest = logger_destination_type_cls()
            destinations.append(logger_dest)
        destinations.append(RemoteUploaderDownloader(bucket_uri='foo'))
    logger = Logger(state=dummy_state, destinations=destinations)
    for logger_destination_property_name in [
            'wandb', 'mlflow', 'mosaicml', 'tensorboard', 'slack', 'console', 'progress_bar', 'file', 'in_memory',
            'remote_uploader_downloader'
    ]:
        assert hasattr(logger, logger_destination_property_name)
        if num_dest_instances_per_type == 0:
            assert getattr(logger, logger_destination_property_name) is None
        else:
            assert getattr(logger, logger_destination_property_name) is not None
            if num_dest_instances_per_type == 1:
                assert not isinstance(getattr(logger, logger_destination_property_name), List)
            else:
                assert isinstance(getattr(logger, logger_destination_property_name), List)


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
