# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import platform

import PIL
import pytest


@pytest.mark.skipif('composer-python' not in os.environ['PATH'] or 'Linux' not in platform.system(),
                    reason='Pillow-simd test only checks if using the composer docker')
class TestDocker:

    def test_pillow_simd(self):
        assert 'post' in PIL.__version__, 'pillow-simd is not installed'
