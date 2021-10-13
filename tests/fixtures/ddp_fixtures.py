# Copyright 2021 MosaicML. All Rights Reserved.

import os
import pathlib
from typing import List

import _pytest.config
import _pytest.config.argparsing
import _pytest.fixtures
import _pytest.mark
import pytest

_MAIN_PYTEST_KEY = "MAIN_PYTEST"


@pytest.fixture
def is_main_pytest_process() -> bool:
    if _MAIN_PYTEST_KEY in os.environ:
        return False
    else:
        os.environ[_MAIN_PYTEST_KEY] = "1"
        return True


@pytest.fixture(autouse=True)
def reset_osenviron(is_main_pytest_process: None):
    # Require the is_main_pytest_process fixture so that will be set before the env variable capture
    original_environ = os.environ.copy()
    try:
        yield
    finally:
        # remove the extra keys
        keys_to_remove: List[str] = []
        for k in os.environ.keys():
            if k not in original_environ:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del os.environ[k]

        # reset the other keys
        for k, v in original_environ.items():
            os.environ[k] = v


@pytest.fixture(autouse=True)
def ddp_fork_multi_pytest(request: _pytest.fixtures.FixtureRequest, reset_osenviron: None) -> None:
    # When using DDP fork, the same command is executed again.
    # When multiple tests are executed with a single pytest command,
    # then the DDP fork will re-run all tests, when instead we just want
    # the forked test to run again. To prevent this behavior, we store the
    # running pytest id as an environ. If this variable is set, and it does
    # NOT match the current pytest test ID, then the test should be skipped
    # as the test was launched by a different DDP fork
    test_name: str = request.node.name
    if "PYTEST_TEST_ID" not in os.environ:
        os.environ["PYTEST_TEST_ID"] = test_name
        return
    if os.environ["PYTEST_TEST_ID"] != test_name:
        pytest.skip("Skipping test invoked within a DDP fork")


@pytest.fixture
def ddp_tmpdir(tmpdir: pathlib.Path) -> str:
    if os.environ.get("DDP_TMP_DIR") is None:
        # Need to set the tmpdir as an environ, so all subprocesses share the same tmpdir.
        # Upon subprocess calls, pytest gives each ddp process a different tmpdir
        # Instead, we want to use the same tmpdir for test case assertions.
        os.environ['DDP_TMP_DIR'] = str(tmpdir)
        return str(tmpdir)
    return os.environ["DDP_TMP_DIR"]
