# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os

from setuptools import setup

setup(
    name="pillow",
    version=os.environ['PILLOW_PSEUDOVERSION'],
)
