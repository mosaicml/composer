# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for creating and loading compressed files."""

import shutil
import subprocess
from contextlib import contextmanager
from typing import IO, Iterator, List, Optional

__all__ = ['is_compressed_pt', 'CliCompressor', 'get_compressor', 'KNOWN_COMPRESSORS']


class CompressorNotFound(FileNotFoundError):
    pass


def is_compressed_pt(filename: str) -> bool:
    """Whether the filename is for a directly compressed pt file.

    Whether the extension of the given filename indicates that the file contains a raw compressed stream
    of a single pt file without a container (like tar).
    """
    parts = filename.split('.')
    return len(parts) >= 2 and parts[-2] == 'pt'


class CliCompressor:
    """Base class for data compression CLI tools."""

    def __init__(self, extension: str, cmd: Optional[str] = None) -> None:
        self.extension = extension
        self.cmd = cmd if cmd is not None else extension

    @property
    def exists(self) -> bool:
        return shutil.which(self.cmd) is not None

    def check_exists(self) -> None:
        if not self.exists:
            raise CompressorNotFound(f'Could not find command "{self.cmd}" in the PATH.')

    def _compress_cmd(self) -> List[str]:
        return [self.cmd]

    @contextmanager
    def compress(self, filename: str) -> Iterator[IO[bytes]]:
        self.check_exists()
        with open(filename, 'wb') as f:
            proc = subprocess.Popen(
                self._compress_cmd(),
                stdin=subprocess.PIPE,
                stdout=f,
            )
            assert proc.stdin is not None
            yield proc.stdin
            proc.stdin.close()
            proc.wait()

    def _decompress_cmd(self, filename: str) -> List[str]:
        return [self.cmd, '-dc', filename]

    @contextmanager
    def decompress(self, in_filename: str) -> Iterator[IO[bytes]]:
        self.check_exists()
        proc = subprocess.Popen(
            self._decompress_cmd(in_filename),
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        yield proc.stdout
        proc.wait()


def get_compressor(filename: str) -> CliCompressor:
    """Obtain the compressor that supports the format of the given file."""
    if not is_compressed_pt(filename):
        raise ValueError(f'The given filename does not correspond to a compressed file: "{filename}".')
    extension = filename.split('.')[-1]
    for c in KNOWN_COMPRESSORS:
        if c.extension == extension:
            return c
    raise CompressorNotFound(f'Could not find compressor for "{filename}".')


KNOWN_COMPRESSORS = [
    CliCompressor('bz2', 'bzip2'),
    CliCompressor('gz', 'gzip'),
    CliCompressor('lz4'),
    CliCompressor('lzma'),
    CliCompressor('lzo', 'lzop'),
    CliCompressor('xz'),
    CliCompressor('zst', 'zstd'),
]
