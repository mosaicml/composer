# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest

from composer.utils.compression import (
    KNOWN_COMPRESSORS,
    CliCompressor,
    CompressorNotFound,
    get_compressor,
    is_compressed_pt,
)


def test_is_compressed_pt() -> None:
    assert is_compressed_pt('x.pt.lz4')
    assert is_compressed_pt('x.pt.unknown')
    assert is_compressed_pt('pt.lz4')
    assert is_compressed_pt('pt.unknown')
    assert is_compressed_pt('x.y.pt.lz4')
    assert is_compressed_pt('x.y.pt.unknown')

    assert not is_compressed_pt('')
    assert not is_compressed_pt('x.lz4')
    assert not is_compressed_pt('x.tar.lz4')


def test_get_invalid_compressor() -> None:
    with pytest.raises(CompressorNotFound, match=re.escape('Could not find compressor for "foo.pt.unknown".')):
        get_compressor('foo.pt.unknown')


def test_compressor_not_found() -> None:
    compressor = CliCompressor('foobar', 'unknown_compressor')
    assert not compressor.exists
    with pytest.raises(CompressorNotFound):
        compressor.check_exists()


@pytest.mark.parametrize('compressor', KNOWN_COMPRESSORS)
def test_compressor(tmp_path: Path, compressor: CliCompressor) -> None:
    if not compressor.exists:
        pytest.skip(reason=f'compressor {compressor.cmd} not found')

    test_file = tmp_path / 'my_file'
    test_data = b'foo foo foo'

    with compressor.compress(str(test_file)) as f:
        f.write(test_data)
    assert test_file.exists()
    assert test_file.read_bytes() != test_data

    with compressor.decompress(str(test_file)) as f:
        assert f.read() == test_data
