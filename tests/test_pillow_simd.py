# Copyright 2021 MosaicML. All Rights Reserved.

import platform

import PIL
import pytest


@pytest.mark.skipif(
    'Linux' not in platform.system(),
    reason="Pillow-SIMD not available on non-Linux systems.",
)
def test_pillow_simd():
    assert "post" in PIL.__version__, "pillow-simd is not installed"
