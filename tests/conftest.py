# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Optional

import pytest

from composer.utils import reproducibility

# Allowed options for pytest.mark.world_size()
# Important: when updating this list, make sure to also up ./.ci/test.sh
# (so tests of all world sizes will be executed) and tests/README.md
# (so the documentation is correct)
WORLD_SIZE_OPTIONS = (1, 2)

# Enforce deterministic mode before any tests start.
reproducibility.configure_deterministic_mode()

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    'tests.fixtures.autouse_fixtures',
    'tests.fixtures.fixtures',
    'tests.fixtures.synthetic_hf_state',
]


def _add_option(parser: pytest.Parser, name: str, help: str, choices: Optional[List[str]] = None):
    parser.addoption(
        f'--{name}',
        default=None,
        type=str,
        choices=choices,
        help=help,
    )
    parser.addini(
        name=name,
        help=help,
        type='string',
        default=None,
    )


def _get_option(config: pytest.Config, name: str, default: Optional[str] = None) -> str:  # type: ignore
    val = config.getoption(name)
    if val is not None:
        assert isinstance(val, str)
        return val
    val = config.getini(name)
    if val == []:
        val = None
    if val is None:
        if default is None:
            pytest.fail(f'Config option {name} is not specified but is required')
        val = default
    assert isinstance(val, str)
    return val


def pytest_addoption(parser: pytest.Parser) -> None:
    _add_option(parser,
                'seed',
                help="""\
        Rank zero seed to use. `reproducibility.seed_all(seed + dist.get_global_rank())` will be invoked
        before each test.""")
    _add_option(parser, 'sftp_uri', help='SFTP URI for integration tests.')
    _add_option(parser, 's3_bucket', help='S3 Bucket for integration tests')


def _get_world_size(item: pytest.Item):
    """Returns the world_size of a test, defaults to 1."""
    _default = pytest.mark.world_size(1).mark
    return item.get_closest_marker('world_size', default=_default).args[0]


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Filter tests by world_size (for multi-GPU tests) and duration (short, long, or all)"""

    world_size = int(os.environ.get('WORLD_SIZE', '1'))

    conditions = [
        lambda item: _get_world_size(item) == world_size,
    ]

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


# Note: These methods are an alternative to the tiny_bert fixtures above.
# Fixtures cannot be used natively as paramaterized inputs, which we require when
# we wish to run a test across multiple models, one of which is a HuggingFace BERT Tiny.
# As a workaround, we inject objects into the PyTest namespace. Tests should not directly
# use pytest.{var}, but instead should import and use the helper copy methods configure_{var}
# so the objects in the PyTest namespace do not change.
def pytest_configure():
    transformers = pytest.importorskip('transformers')
    config = transformers.AutoConfig.from_pretrained('prajjwal1/bert-tiny')
    hf_model = transformers.AutoModelForMaskedLM.from_config(config)  # type: ignore (thirdparty)
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    pytest.tiny_bert_model = hf_model  # type: ignore
    pytest.tiny_bert_tokenizer = hf_tokenizer  # type: ignore
    pytest.tiny_bert_config = config  # type: ignore


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):
    if exitstatus == 5:
        session.exitstatus = 0  # Ignore no-test-ran errors
