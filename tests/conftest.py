# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import pathlib
from typing import List, Optional

import pytest
import torch
import tqdm.std

import composer
from composer.utils import dist, reproducibility

# Allowed options for pytest.mark.world_size()
# Important: when updating this list, make sure to also up ./.ci/test.sh
# (so tests of all world sizes will be executed) and tests/README.md
# (so the documentation is correct)
WORLD_SIZE_OPTIONS = (1, 2)

# default timout threshold is 2 seconds for determinign long and short
DEFAULT_TIMEOUT = 2.0

# Enforce deterministic mode before any tests start.
reproducibility.configure_deterministic_mode()

# during the pytest refactor transition, this flag
# indicates whether to include the deprecated fixtures.
# used for internal development.
_include_deprecated_fixtures = True

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    'tests.fixtures.new_fixtures',
    'tests.fixtures.synthetic_hf_state',
]

if _include_deprecated_fixtures:
    pytest_plugins += [
        'tests.fixtures.dummy_fixtures',
    ]

if torch.cuda.is_available():
    # torch.cuda takes a few seconds to initialize on first load
    # pre-initialize cuda so individual tests, which are subject to a timeout,
    # load cuda instantly.
    torch.Tensor([0]).cuda()


def _add_option(parser: pytest.Parser,
                name: str,
                help: str,
                default: Optional[str] = None,
                choices: Optional[List[str]] = None):
    parser.addoption(
        f'--{name}',
        default=default,
        type=str,
        choices=choices,
        help=help,
    )
    parser.addini(
        name=name,
        help=help,
        type='string',
        default=default,
    )


def _get_option(config: pytest.Config, name: str, skip: bool = False) -> Optional[str]:
    val = config.getoption(name)
    if val is not None:
        assert isinstance(val, str)
        return val
    val = config.getini(name)
    if val is None and skip:
        pytest.skip(f'Config option {name} is not specified; skipping test')
    assert val is None or isinstance(val, str)
    return val


def pytest_addoption(parser: pytest.Parser) -> None:
    _add_option(parser,
                'seed',
                default='0',
                help="""\
        Rank zero seed to use. `reproducibility.seed_all(seed + dist.get_global_rank())` will be invoked
        before each test.""")
    _add_option(parser,
                'duration',
                default='all',
                choices=['short', 'long', 'all'],
                help="""Duration of tests, one of short, long, or all.
                             Tests are short if their timeout < 2 seconds
                             (configurable threshold). Default: all.""")
    _add_option(parser, 'sftp_uri', help='SFTP URI for integration tests.')
    _add_option(parser, 's3_bucket', help='S3 Bucket for integration tests')


def _get_timeout(item: pytest.Item, default: float):
    """Returns the timeout of a test, defaults to -1."""
    _default = pytest.mark.timeout(default).mark
    timeout = item.get_closest_marker('timeout', default=_default).args[0]
    return float('inf') if timeout == 0 else timeout  # timeout(0) means no timeout restrictions


def _get_world_size(item: pytest.Item):
    """Returns the world_size of a test, defaults to 1."""
    _default = pytest.mark.world_size(1).mark
    return item.get_closest_marker('world_size', default=_default).args[0]


def _validate_duration(duration: Optional[str]):
    if duration not in ('short', 'long', 'all'):
        raise ValueError(f'duration ({duration}) must be one of short, long, all.')


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Filter tests by world_size (for multi-GPU tests) and duration (short, long, or all)"""
    threshold = float(getattr(config, '_env_timeout', DEFAULT_TIMEOUT))
    duration = _get_option(config, 'duration')
    assert isinstance(duration, str)
    _validate_duration(duration)

    world_size = int(os.environ.get('WORLD_SIZE', '1'))

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
def rank_zero_seed(pytestconfig: pytest.Config) -> int:
    """Read the rank_zero_seed from the CLI option."""
    seed = _get_option(pytestconfig, 'seed')
    assert seed is not None
    return int(seed)


@pytest.fixture(autouse=True)
def seed_all(rank_zero_seed: int, monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch reproducibility get_random_seed to always return the rank zero seed, and set the random seed before
    each test to the rank local seed."""
    monkeypatch.setattr(reproducibility, 'get_random_seed', lambda: rank_zero_seed)
    reproducibility.seed_all(rank_zero_seed + dist.get_global_rank())


@pytest.fixture(autouse=True)
def chdir_to_tmp_path(tmp_path: pathlib.Path):
    os.chdir(tmp_path)


@pytest.fixture(autouse=True, scope='session')
def disable_tqdm_bars():
    # Disable tqdm progress bars globally in tests
    original_tqdm_init = tqdm.std.tqdm.__init__

    def new_tqdm_init(*args, **kwargs):
        if 'disable' not in kwargs:
            kwargs['disable'] = True
        return original_tqdm_init(*args, **kwargs)

    # Not using pytest monkeypatch as it is a function-scoped fixture
    tqdm.std.tqdm.__init__ = new_tqdm_init


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if exitstatus == 5:
        session.exitstatus = 0  # Ignore no-test-ran errors


@pytest.fixture
def sftp_uri(pytestconfig: pytest.Config):
    return _get_option(pytestconfig, 'sftp_uri', skip=True)


@pytest.fixture
def s3_bucket(pytestconfig: pytest.Config):
    return _get_option(pytestconfig, 's3_bucket', skip=True)
