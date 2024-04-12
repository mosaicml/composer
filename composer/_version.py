# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The Composer Version."""

import os
import subprocess

__version__ = '0.21.2'

# Add the git sha to the version if available
cwd = os.path.dirname(os.path.abspath(__file__))
sha = None
try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    __version__ += '+' + sha[:7]
except Exception:
    pass
