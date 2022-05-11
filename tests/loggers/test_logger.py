# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest

from composer.core.state import State
from composer.loggers import Logger, LoggerDestination, LogLevel
from composer.utils import dist, reproducibility


@pytest.mark.world_size(2)
def test_logger_run_name(dummy_state: State):
    # seeding with the global rank to ensure that each rank has a different seed
    reproducibility.seed_all(dist.get_global_rank())

    logger = Logger(state=dummy_state)
    # The run name should be the same on every rank -- it is set via a distributed reduction
    # Manually verify that all ranks have the same run name
    run_names = dist.all_gather_object(logger.run_name)
    assert len(run_names) == 2  # 2 ranks
    assert all(run_name == run_names[0] for run_name in run_names)


def test_logger_file_artifact(dummy_state: State):

    file_logged = False

    class DummyLoggerDestination(LoggerDestination):

        def log_file_artifact(self, state: State, log_level: LogLevel, artifact_name: str, file_path: pathlib.Path, *,
                              overwrite: bool):
            nonlocal file_logged
            file_logged = True
            assert artifact_name == "foo"
            assert file_path.name == "bar"
            assert overwrite

    logger = Logger(state=dummy_state, destinations=[DummyLoggerDestination()])
    logger.file_artifact(
        log_level="epoch",
        artifact_name="foo",
        file_path="bar",
        overwrite=True,
    )

    assert file_logged
