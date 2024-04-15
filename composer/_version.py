# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Composer Version."""

import inspect
import os
import subprocess

__version__ = '0.21.2'

# Add the git sha to the version if available
file_path = inspect.getfile(lambda: None)  # more robust than __file__
cwd = os.path.dirname(file_path)
sha = None
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    __version__ += '+' + sha[:7]
except Exception:
    pass
