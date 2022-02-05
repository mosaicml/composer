# Copyright 2021 MosaicML. All Rights Reserved.

import logging
import os
import pathlib
from typing import List, Optional

import _pytest.config
import _pytest.config.argparsing
import _pytest.fixtures
import _pytest.mark
import pytest
from _pytest.monkeypatch import MonkeyPatch

import composer
from composer.utils import run_directory

# Allowed options for pytest.mark.world_size()
# Important: when updating this list, make sure to also up scripts/test.sh
# so tests of all world sizes will be executed
WORLD_SIZE_OPTIONS = (1, 2)

# Set this before running any tests, since it won't take effect if there are any cudnn operations
# in a previous test and then this variable is set by a latter test
# see composer.utils.reproducibility.configure_deterministic_mode
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    "tests.fixtures.dummy_fixtures",
    "tests.fixtures.distributed_fixtures",
    "tests.algorithms.algorithm_fixtures",
]


def pytest_addoption(parser: _pytest.config.argparsing.Parser) -> None:
    parser.addoption("--duration",
                     default="short",
                     choices=["short", "long", "all"],
                     help="""Duration of tests, one of short, long, or all.
                             Tests are short if their timeout < 2 seconds
                             (configurable threshold). Default: short.""")


def _get_test_world_size(item: pytest.Item) -> int:
    world_size_marks = tuple(item.iter_markers(name="world_size"))
    assert len(world_size_marks) <= 1
    if len(world_size_marks) == 1:
        world_size_mark = world_size_marks[0]
        return world_size_mark.args[0]
    else:
        return 1


def _get_timeout(item: pytest.Item) -> Optional[float]:
    # see if it has a timeout marker
    timeout_mark = item.get_closest_marker("timeout")
    if timeout_mark is None:
        return None
    return timeout_mark.args[0]


def _filter_items_for_world_size(items: List[pytest.Item]) -> None:
    items_to_remove: List[pytest.Item] = []
    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    for item in items:
        test_world_size = _get_test_world_size(item)
        if test_world_size not in WORLD_SIZE_OPTIONS:
            name = item.name
            raise ValueError(
                f"Invalid option @pytest.mark.world_size({test_world_size}) for test `{name}`: @pytest.mark.world_size() must be set to one of {WORLD_SIZE_OPTIONS}"
            )
        else:
            # if the number of gpus is explicitley specified,
            # then ensure that it matches
            if env_world_size != test_world_size:
                items_to_remove.append(item)

    for item in items_to_remove:
        items.remove(item)


def _filter_items_for_timeout(config: _pytest.config.Config, items: List[pytest.Item]):
    default_timeout = getattr(config, "_env_timeout")
    assert isinstance(default_timeout, float), "should be set by the toml/ini"
    cli_test_duration = config.getoption("duration", default=None)
    assert cli_test_duration is not None, "should be set by argparse"
    if cli_test_duration == "all":
        return
    items_to_remove: List[pytest.Item] = []
    for item in items:
        test_timeout = _get_timeout(item)
        if test_timeout is None:
            test_timeout = default_timeout
        is_long_test = test_timeout == 0 or test_timeout > default_timeout
        if is_long_test != (cli_test_duration == "long"):
            items_to_remove.append(item)
    for item in items_to_remove:
        items.remove(item)


def pytest_collection_modifyitems(session: pytest.Session, config: _pytest.config.Config,
                                  items: List[pytest.Item]) -> None:
    del session  # unused
    _filter_items_for_world_size(items)
    _filter_items_for_timeout(config, items)


@pytest.fixture(autouse=True)
def set_loglevels():
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
