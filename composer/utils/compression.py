# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for creating and loading compressed files."""

import shutil
import subprocess
from contextlib import contextmanager
from typing import IO, Iterator, Optional

__all__ = ['is_compressed_pt', 'CliCompressor', 'get_compressor', 'KNOWN_COMPRESSORS']


class CompressorNotFound(FileNotFoundError):
    pass


def is_compressed_pt(filename: str) -> bool:
    """Whether the filename is for a directly compressed pt file.

    Whether the extension of the given filename indicates that the file contains a raw compressed stream
    of a single pt file without a container (like tar).
    """
    parts = filename.split('.')
    return len(parts) >= 2 and parts[-2] == 'pt' and parts[-1] != 'symlink'


class CliCompressor:
    """Base class for data compression CLI tools.

    This class handles compression and decompression of data by piping it through
    CLI compressor tools installed on the system. e.g. the `gzip` command for producing `.gz` files.

    Example:
    .. code-block:: python

        compressor = CliCompressor('gz', 'gzip')

        with compressor.compress('myfile.txt.gz') as f:
            f.write('foo')

        with compressor.decompress('myfile.txt.gz') as f:
            assert f.read() == 'foo'

    Args:
        extension (str): The suffix used to identify files that the compressor supports (without a leading `.`).
        cmd (str, optional): The name of the CLI tool that this compressor uses. Defaults to `None`, in which case
            it is assumed that the tool name is the same as the extension.
    """

    def __init__(self, extension: str, cmd: Optional[str] = None) -> None:
        self.extension = extension
        self.cmd = cmd if cmd is not None else extension

    def __repr__(self) -> str:
        return f'CliCompressor({self.extension!r}, {self.cmd!r})'

    @property
    def exists(self) -> bool:
        """Whether the CLI tool used by this compressor can be found."""
        return shutil.which(self.cmd) is not None

    def check_exists(self) -> None:
        if not self.exists:
            raise CompressorNotFound(f'Could not find command "{self.cmd}" in the PATH.')

    def _compress_cmd(self) -> list[str]:
        return [self.cmd]

    @contextmanager
    def compress(self, out_filename: str) -> Iterator[IO[bytes]]:
        """Compress some data, saving to the given file."""
        self.check_exists()
        with open(out_filename, 'wb') as f:
            proc = subprocess.Popen(
                self._compress_cmd(),
                stdin=subprocess.PIPE,
                stdout=f,
            )
            assert proc.stdin is not None
            yield proc.stdin
            proc.stdin.close()
            returncode = proc.wait()
            if returncode != 0:
                raise IOError(f'failed to compress to "{out_filename}" using {self!r} (return code {returncode})')

    def _decompress_cmd(self, filename: str) -> list[str]:
        return [self.cmd, '-dc', filename]

    @contextmanager
    def decompress(self, in_filename: str) -> Iterator[IO[bytes]]:
        """Decompress the content of the given file, providing the output as a file-like object."""
        self.check_exists()
        proc = subprocess.Popen(
            self._decompress_cmd(in_filename),
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        yield proc.stdout
        returncode = proc.wait()
        if returncode != 0:
            raise IOError(f'failed to decompress "{in_filename}" using {self!r} (return code {returncode})')


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
