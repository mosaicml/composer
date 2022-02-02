# Copyright 2021 MosaicML. All Rights Reserved.

import PIL


def test_pillow_simd():
    assert "post" in PIL.__version__, "pillow-simd is not installed"
