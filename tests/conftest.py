# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import pathlib
from typing import List, Optional

import pytest
from pytest import MonkeyPatch

import composer
from composer.utils import dist, reproducibility, run_directory

# Allowed options for pytest.mark.world_size()
# Important: when updating this list, make sure to also up ./.ci/test.sh
# (so tests of all world sizes will be executed) and tests/README.md
# (so the documentation is correct)
WORLD_SIZE_OPTIONS = (1, 2)

# default timout threshold is 2 seconds for determinign long and short
DEFAULT_TIMEOUT = 2.0

# Enforce use of deterministic kernels
# see composer.utils.reproducibility.configure_deterministic_mode
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# during the pytest refactor transition, this flag
# indicates whether to include the deprecated fixtures.
# used for internal development.
_include_deprecated_fixtures = True

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    "tests.fixtures.new_fixtures",
]

if _include_deprecated_fixtures:
    pytest_plugins += [
        "tests.fixtures.dummy_fixtures",
        "tests.fixtures.distributed_fixtures",
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--seed",
                     default=0,
                     type=int,
                     help="""\
        Rank zero seed to use. `reproducibility.seed_all(seed + dist.get_global_rank())` will be invoked
        before each test.""")
    parser.addoption("--duration",
                     default="all",
                     choices=["short", "long", "all"],
                     help="""Duration of tests, one of short, long, or all.
                             Tests are short if their timeout < 2 seconds
                             (configurable threshold). Default: all.""")
    parser.addoption("--world-size",
                     default=int(os.environ.get('WORLD_SIZE', 1)),
                     type=int,
                     choices=WORLD_SIZE_OPTIONS,
                     help="""Filters the tests based on their requested world size.
                             Defaults to 1, and can also be set by the WORLD_SIZE
                             environment variable. For world_size>1, please launch
                             with the composer launcher.""")


def _get_timeout(item: pytest.Item, default: float):
    """Returns the timeout of a test, defaults to -1."""
    _default = pytest.mark.timeout(default).mark
    timeout = item.get_closest_marker("timeout", default=_default).args[0]
    return float('inf') if timeout == 0 else timeout  # timeout(0) means no timeout restrictions


def _get_world_size(item: pytest.Item):
    """Returns the world_size of a test, defaults to 1."""
    _default = pytest.mark.world_size(1).mark
    return item.get_closest_marker("world_size", default=_default).args[0]


def _validate_world_size(world_size: Optional[int]):
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) != world_size:
        raise ValueError(f'--world-size ({world_size}) and WORLD_SIZE environment'
                         f'variable ({os.environ["WORLD_SIZE"]}) do not match.')


def _validate_duration(duration: Optional[str]):
    if duration not in ('short', 'long', 'all'):
        raise ValueError(f'duration ({duration}) must be one of short, long, all.')


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Filter tests by world_size (for multi-GPU tests) and duration (short, long, or all)"""
    threshold = float(getattr(config, "_env_timeout", DEFAULT_TIMEOUT))
    duration = config.getoption("duration")
    world_size = config.getoption("world_size")

    assert isinstance(duration, str)
    assert isinstance(world_size, int)

    _validate_world_size(world_size)
    _validate_duration(duration)

    conditions = [
        lambda item: _get_world_size(item) == world_size,
    ]

    # separate tests by whether timeout is < or > threshold.
    if duration == 'short':
        conditions += [lambda item: _get_timeout(item, default=threshold) <= threshold]
    elif duration == 'long':
        conditions += [lambda item: _get_timeout(item, default=threshold) > threshold]

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


@pytest.fixture
def rank_zero_seed(request: pytest.FixtureRequest) -> int:
    """Read the rank_zero_seed from the CLI option."""
    seed = request.config.getoption("seed")
    assert isinstance(seed, int)
    return seed


@pytest.fixture(autouse=True)
def seed_all(rank_zero_seed: int):
    """Set the random seed before each test to ensure consistent test results, which and limit flakiness due to random
    initializations."""
    reproducibility.seed_all(rank_zero_seed + dist.get_global_rank())


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
