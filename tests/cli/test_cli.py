# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from typing import List

import pytest

import composer


@pytest.mark.parametrize('args', [
    ['composer', '--version'],
    ['python', '-m', 'composer', '--version'],
    ['python', '-m', 'composer.cli', '--version'],
    ['python', '-m', 'composer.cli.launcher', '--version'],
])
def test_cli_version(args: List[str]):
    version_str = subprocess.check_output(args, text=True)
    print(version_str)
    print(subprocess.check_output(['pip', 'list'], text=True))
    print(subprocess.check_output(['which', 'composer'], text=True))
    print(subprocess.check_output(['which', 'python'], text=True))
    print(f'Composer code: {composer.__version__}')
    assert version_str == f'MosaicML Composer {composer.__version__}\n'
