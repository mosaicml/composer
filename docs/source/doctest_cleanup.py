# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

# variables are defined in doctest_fixtures.py
# pyright: reportUndefinedVariable=none

# tmpdir and cwd were defined in doctest_fixtures.py

os.chdir(cwd)

tmpdir.cleanup()
