# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Cleanup script that is executed at the end of each doctest."""

import os
import shutil

# variables are defined in doctest_fixtures.py
# pyright: reportUndefinedVariable=none

# tmpdir and cwd were defined in doctest_fixtures.py

os.chdir(cwd)

try:
    shutil.rmtree(tmpdir)
except OSError:
    pass
