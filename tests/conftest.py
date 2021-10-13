# Copyright 2021 MosaicML. All Rights Reserved.

import atexit
from typing import Callable, List

import _pytest.config
import _pytest.config.argparsing
import _pytest.fixtures
import _pytest.mark
import pytest
from _pytest.monkeypatch import MonkeyPatch

N_GPU_OPTIONS = (0, 1, 2, 4, 8)

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    "tests.fixtures.dummy_fixtures",
    "tests.fixtures.ddp_fixtures",
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


def _get_n_gpus(item: pytest.Item) -> int:
    n_gpu_marks = tuple(item.iter_markers(name="n_gpus"))
    assert len(n_gpu_marks) <= 1
    if len(n_gpu_marks) == 1:
        n_gpu_mark = n_gpu_marks[0]
        return n_gpu_mark.args[0]
    else:
        return 0


def pytest_collection_modifyitems(session: pytest.Session, config: _pytest.config.Config,
                                  items: List[pytest.Item]) -> None:
    items_to_remove: List[pytest.Item] = []
    cli_n_gpus = config.getoption("n_gpus", default=None)
    if cli_n_gpus is None:
        return
    for item in items:
        test_n_gpus = _get_n_gpus(item)
        if cli_n_gpus != test_n_gpus:
            items_to_remove.append(item)
        if test_n_gpus not in N_GPU_OPTIONS:
            name = item.name
            raise ValueError(
                f"Invalid option @pytest.mark.n_gpus({test_n_gpus}) for test `{name}`: @pytest.mark.n_gpus() must be set to one of {N_GPU_OPTIONS}"
            )
    for item in items_to_remove:
        items.remove(item)


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
