# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import pathlib
from typing import List

import pytest
from pytest import MonkeyPatch

import composer
from composer.utils import run_directory

# Allowed options for pytest.mark.world_size()
# Important: when updating this list, make sure to also up scripts/test.sh
# so tests of all world sizes will be executed
WORLD_SIZE_OPTIONS = (1, 2)

# Enforce use of deterministic kernels
# see composer.utils.reproducibility.configure_deterministic_mode
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# during the pytest refactor transition, this flag
# indicates whether to include the deprecated fixtures
include_deprecated_fixtures = True

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    "tests.fixtures.new_fixtures",
]

if include_deprecated_fixtures:
    pytest_plugins += [
        "tests.fixtures.dummy_fixtures",
        "tests.fixtures.distributed_fixtures",
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--duration",
                     default="short",
                     choices=["short", "long", "all"],
                     help="""Duration of tests, one of short, long, or all.
                             Tests are short if their timeout < 2 seconds
                             (configurable threshold). Default: short.""")
    parser.addoption("--world-size",
                     default=int(os.environ.get('WORLD_SIZE', 1)),
                     type=int,
                     choices=WORLD_SIZE_OPTIONS,
                     help="""Number of devices. Filters the tests based on their
                            requested world size. Defaults to 1, and can also
                            be set by the WORLD_SIZE environment variable.""")


def _get_timeout(item: pytest.Item):
    """Returns the timeout of a test, defaults to 0."""
    _default = pytest.mark.timeout(0).mark
    return item.get_closest_marker("timeout", default=_default).args[0]


def _get_world_size(item: pytest.Item):
    """Returns the world_size of a test, defaults to 1."""
    _default = pytest.mark.world_size(1).mark
    return item.get_closest_marker("world_size", default=_default).args[0]


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Filter tests by world_size (for multi-GPU tests) and duration (short, long, or all)"""
    timeout_threshold = getattr(config, "_env_timeout", 2.0)
    duration = config.getoption("duration")
    world_size = config.getoption("world_size")

    conditions = [
        lambda item: _get_world_size(item) == world_size,
    ]

    # separate tests by whether timeout is < or > threshold.
    if duration == 'short':
        conditions += [lambda item: _get_timeout(item) < timeout_threshold]
    elif duration == 'long':
        conditions += [lambda item: _get_timeout(item) > timeout_threshold]

    # keep items that satisfy all conditions
    remaining = []
    deselected = []
    for item in items:
        if all([condition(item) for condition in conditions]):
            remaining.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining


@pytest.fixture(autouse=True)
def set_loglevels():
    """Ensures all log levels are set to DEBUG."""
    logging.basicConfig()
    logging.getLogger(composer.__name__).setLevel(logging.DEBUG)


@pytest.fixture(autouse=True)
def subfolder_run_directory(tmpdir: pathlib.Path, monkeypatch: MonkeyPatch) -> None:
    tmpdir_test_folder_name = os.path.basename(os.path.normpath(str(tmpdir)))
    test_folder_tmpdir = os.path.join(run_directory.get_node_run_directory(), tmpdir_test_folder_name)
    monkeypatch.setenv(run_directory._RUN_DIRECTORY_KEY, test_folder_tmpdir)
    os.makedirs(run_directory.get_run_directory(), exist_ok=True)
    os.makedirs(run_directory.get_node_run_directory(), exist_ok=True)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if exitstatus == 5:
        session.exitstatus = 0  # Ignore no-test-ran errors
