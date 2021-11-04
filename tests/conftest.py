# Copyright 2021 MosaicML. All Rights Reserved.

import atexit
import logging
from typing import Callable, List, Optional

import _pytest.config
import _pytest.config.argparsing
import _pytest.fixtures
import _pytest.mark
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

import composer

N_GPU_OPTIONS = (0, 1, 2, 4, 8)

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    "tests.fixtures.dummy_fixtures",
    "tests.algorithms.algorithm_fixtures",
]


def pytest_addoption(parser: _pytest.config.argparsing.Parser) -> None:
    parser.addoption(
        "--n_gpus",
        default=None,
        type=int,
        choices=N_GPU_OPTIONS,
        help=
        "Number of GPUs available. Only tests marked with @pytest.mark.n_gpus(val) will be executed. Defaults to all tests."
    )

    parser.addoption(
        "--test_duration",
        default="short",
        choices=["short", "long", "all"],
        help=
        "Whether to run short tests (the default), long tests, or all tests. A test is considered short if its timeout is "
        "less than that specified in the config.")


def _get_n_gpus(item: pytest.Item) -> int:
    n_gpu_marks = tuple(item.iter_markers(name="n_gpus"))
    assert len(n_gpu_marks) <= 1
    if len(n_gpu_marks) == 1:
        n_gpu_mark = n_gpu_marks[0]
        return n_gpu_mark.args[0]
    else:
        return 0


def _get_timeout(item: pytest.Item) -> Optional[float]:
    # see if it has a timeout marker
    timeout_mark = item.get_closest_marker("timeout")
    if timeout_mark is None:
        return None
    return timeout_mark.args[0]


def _filter_items_for_n_gpus(config: _pytest.config.Config, items: List[pytest.Item]) -> None:
    items_to_remove: List[pytest.Item] = []
    cli_n_gpus = config.getoption("n_gpus", default=None)
    for item in items:
        test_n_gpus = _get_n_gpus(item)
        if test_n_gpus not in N_GPU_OPTIONS:
            name = item.name
            raise ValueError(
                f"Invalid option @pytest.mark.n_gpus({test_n_gpus}) for test `{name}`: @pytest.mark.n_gpus() must be set to one of {N_GPU_OPTIONS}"
            )
        if cli_n_gpus is None:
            # run if we have gpus available
            if test_n_gpus > torch.cuda.device_count():
                items_to_remove.append(item)
                continue
        else:
            # if the number of gpus is explicitley specified,
            # then ensure that tit matches
            if cli_n_gpus != test_n_gpus:
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
    _filter_items_for_n_gpus(config, items)
    _filter_items_for_timeout(config, items)


def parameterize_n_gpus(*n_gpus: int, field_name: str = "n_gpus") -> Callable[..., Callable[..., None]]:

    def wrapped_func(testfunc: Callable[..., None]) -> Callable[..., None]:
        return pytest.mark.parametrize(
            field_name, [pytest.param(n_gpu, marks=[pytest.mark.n_gpus(n_gpu)]) for n_gpu in n_gpus])(testfunc)

    return wrapped_func


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
