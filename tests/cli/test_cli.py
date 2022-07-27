# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from typing import List

import pytest

import composer


@pytest.mark.daily
@pytest.mark.parametrize('args', [
    ['composer', '--version'],
    [sys.executable, '-m', 'composer', '--version'],
    [sys.executable, '-m', 'composer.cli', '--version'],
    [sys.executable, '-m', 'composer.cli.launcher', '--version'],
])
def test_cli_version(args: List[str]):
    version_str = subprocess.check_output(args, text=True)
    assert version_str == f'composer {composer.__version__}\n'
