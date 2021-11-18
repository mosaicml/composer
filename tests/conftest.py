# Copyright 2021 MosaicML. All Rights Reserved.

import atexit
import datetime
import logging
import os
import pathlib
import time
from typing import List, Optional

import _pytest.config
import _pytest.config.argparsing
import _pytest.fixtures
import _pytest.mark
import pytest
import torch.distributed
from _pytest.monkeypatch import MonkeyPatch

import composer
from composer.utils.run_directory import get_relative_to_run_directory

# Allowed options for pytest.mark.world_size()
# Important: when updating this list, make sure to also up scripts/test.sh
# so tests of all world sizes will be executed
WORLD_SIZE_OPTIONS = (1, 2)

PYTEST_DDP_LOCKFILE_DIR = get_relative_to_run_directory(os.path.join("pytest_lockfiles"))
DDP_TIMEOUT = datetime.timedelta(seconds=5)

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    "tests.fixtures.dummy_fixtures",
    "tests.algorithms.algorithm_fixtures",
]


def pytest_addoption(parser: _pytest.config.argparsing.Parser) -> None:
    parser.addoption(
        "--test_duration",
        default="short",
        choices=["short", "long", "all"],
        help=
        "Whether to run short tests (the default), long tests, or all tests. A test is considered short if its timeout is "
        "less than that specified in the config.")


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
    cli_test_duration = config.getoption("test_duration", default=None)
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
def atexit_at_test_end(monkeypatch: MonkeyPatch):
    # monkeypatch atexit so it is called when a test exits, not when the python process exits
    atexit_callbacks = []

    def register(func, *args, **kwargs):
        atexit_callbacks.append((func, args, kwargs))

    monkeypatch.setattr(atexit, "register", register)
    yield
    for func, args, kwargs in atexit_callbacks:
        func(*args, **kwargs)


@pytest.fixture(autouse=True)
def set_loglevels():
    logging.basicConfig()
    logging.getLogger(composer.__name__).setLevel(logging.DEBUG)


@pytest.fixture
def ddp_tmpdir(tmpdir: pathlib.Path) -> str:
    tmpdir_test_folder_name = os.path.basename(os.path.normpath(str(tmpdir)))
    test_folder_tmpdir = get_relative_to_run_directory(tmpdir_test_folder_name, base=os.path.join(str(tmpdir), ".."))
    os.makedirs(test_folder_tmpdir, exist_ok=True)
    return os.path.abspath(test_folder_tmpdir)


@pytest.fixture(autouse=True)
def configure_ddp(request: pytest.FixtureRequest):
    backend = None
    for item in request.session.items:
        marker = item.get_closest_marker('gpu')
        if marker is not None:
            backend = "nccl"
        else:
            backend = "gloo"
        break
    if not torch.distributed.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            torch.distributed.init_process_group(backend, timeout=DDP_TIMEOUT)
        else:
            store = torch.distributed.HashStore()
            torch.distributed.init_process_group(backend, timeout=DDP_TIMEOUT, store=store, world_size=1, rank=0)


@pytest.fixture(autouse=True)
def wait_for_all_procs(ddp_tmpdir: str):
    yield
    if not 'RANK' in os.environ:
        # Not running in a DDP environment
        return
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    proc_lockfile = os.path.join(ddp_tmpdir, f"{global_rank}_finished")
    pathlib.Path(proc_lockfile).touch(exist_ok=False)
    # other processes shouldn't be (too) far behind the current one
    end_time = datetime.datetime.now() + datetime.timedelta(seconds=15)
    for rank in range(world_size):
        if not os.path.exists(os.path.join(ddp_tmpdir, f"{rank}_finished")):
            # sleep for the other procs to write their finished file
            if datetime.datetime.now() < end_time:
                time.sleep(0.1)
            else:
                test_name = os.path.basename(os.path.normpath(ddp_tmpdir))
                raise RuntimeError(f"Rank {rank} did not finish test {test_name}")


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 5:
        session.exitstatus = 0  # Ignore no-test-ran errors
