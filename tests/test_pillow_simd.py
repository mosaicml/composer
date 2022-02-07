# Copyright 2021 MosaicML. All Rights Reserved.

import os
import platform

import PIL
import pytest


@pytest.mark.skipif('composer-python' not in os.environ["PATH"] or 'Linux' not in platform.system(),
                    reason="Pillow-simd test only checks if using the composer docker")
def test_pillow_simd():
    assert "post" in PIL.__version__, "pillow-simd is not installed"
