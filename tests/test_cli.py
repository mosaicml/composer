# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

import composer
from composer.utils.collect_env import print_env


@pytest.mark.parametrize(
    'args',
    [
        ['composer', '--version'],
        ['python', '-m', 'composer', '--version'],
        ['python', '-m', 'composer.cli', '--version'],
        ['python', '-m', 'composer.cli.launcher', '--version'],
    ],
)
def test_cli_version(args: list[str]):
    version_str = subprocess.check_output(args, text=True)
    assert version_str == f'MosaicML Composer {composer.__version__}\n'


def test_collect_env():
    print_env()
