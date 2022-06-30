# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

""":class:`StreamingDatasetWriter` is used to convert a list of samples into binary `.mds` files that can be read as a :class:`StreamingDataset`.
"""

import gzip as gz
import os
import urllib.parse
from io import BufferedWriter
from types import TracebackType
from typing import Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import urllib3
from tqdm import tqdm

from composer.datasets.streaming.format import (StreamingDatasetIndex, get_compression_scheme_basename,
                                                get_index_basename, get_shard_basename, sample_dict_to_bytes)
from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore
from composer.utils.object_store.sftp_object_store import SFTPObjectStore

__all__ = ['StreamingDatasetWriter']


def _parse_compression_args(compression: Optional[str]) -> Tuple[Optional[str], int]:
    """Sets compression settings for the given compression algorithm

    Args:
        compression (str, optional): Compression algorithm and optional compression level. Currently supported: 'gz', 'gz:[1-9]' or None.
    """
    if compression is None:
        return None, 0

    elif compression.startswith('gz'):
        default_compression_level = 6
        compression += f':{default_compression_level}'
        return 'gz', int(compression.split(':')[1])

    else:
        raise NotImplementedError('Unknown compression algorithm')


class StreamingDatasetWriter(object):
    """
    Used for writing a :class:`StreamingDataset` from a list of samples.

    Samples are expected to be of type: ``Dict[str, bytes]``.

    Given each sample, :class:`StreamingDatasetWriter` only writes out the values for a subset of keys (``fields``)
    that are globally shared across the dataset.

    :class:`StreamingDatasetWriter` automatically shards the dataset such that each shard is of size <=
    ``shard_size_limit`` bytes.

    .. doctest::

        To write the dataset:
        >>> from composer.datasets.streaming import StreamingDatasetWriter
        >>> samples = [
        ...     {
        ...         "uid": f"{ix:06}".encode("utf-8"),
        ...         "data": (3 * ix).to_bytes(4, "big"),
        ...         "unused": "blah".encode("utf-8"),
        ...     }
        ...     for ix in range(100)
        ... ]
        >>> dirname = "remote"
        >>> fields = ["uid", "data"]
        >>> with StreamingDatasetWriter(dirname=dirname, fields=fields) as writer:
        ...     writer.write_samples(samples=samples)

        To read the dataset:
        >>> from composer.datasets.streaming import StreamingDataset
        >>> remote = "remote"
        >>> local = "local"
        >>> decoders = {
        ...     "uid": lambda uid_bytes: uid_bytes.decode("utf-8"),
        ...     "data": lambda data_bytes: int.from_bytes(data_bytes, "big"),
        ... }
        >>> dataset = StreamingDataset(remote=remote, local=local, shuffle=False, decoders=decoders)


    Args:
        dirname (str): Directory to write shards to.
        fields: (List[str]): The fields to save for each sample.
        shard_size_limit (int): Maximum shard size in bytes. Default: ``1 << 24``.
        compression (str, optional): Compression algorithm and optional compression level. Currently supported: 'gz', 'gz:[1-9]' or None. Defaults to ``None``.
    """

    default_compression = None

    def __init__(self,
                 dirname: str,
                 fields: List[str],
                 shard_size_limit: int = 1 << 24,
                 compression: Optional[str] = default_compression,
                 remote: Union[ObjectStore, str, None] = None,
                 timeout: Optional[float] = None) -> None:
        if len(fields) != len(set(fields)):
            raise ValueError(f'fields={fields} must be unique.')
        if shard_size_limit <= 0:
            raise ValueError(f'shard_size_limit={shard_size_limit} must be positive.')

        self.dirname = dirname
        os.makedirs(self.dirname, exist_ok=True)
        self.fields = fields
        self.shard_size_limit = shard_size_limit

        # Stats about shards written so far.
        self.samples_per_shard = []
        self.bytes_per_shard = []
        self.bytes_per_sample = []

        # Data of the shard in progress.
        self.new_samples = []
        self.new_shard_size = 0

        # compression scheme for shards
        self.compression_scheme, self.compression_level = _parse_compression_args(compression)

        self.remote = _parse_remote(remote, timeout)

    def _create_binary_file(self, fname: str) -> Union[BufferedWriter, gz.GzipFile]:
        """opens a (potentially compressed) file in binary mode"""

        if self.compression_scheme == 'gz':
            return gz.open(fname, 'xb', compresslevel=self.compression_level)
        elif self.compression_scheme == None:
            return open(fname, 'xb')
        else:
            raise NotImplementedError('unknown compression algorithm')

    def _flush_shard(self) -> None:
        """Flush cached samples to a new dataset shard."""
        shard = len(self.samples_per_shard)
        basename = get_shard_basename(shard, compression_name=self.compression_scheme)
        filename = os.path.join(self.dirname, basename)

        with self._create_binary_file(filename) as out:
            for data in self.new_samples:
                out.write(data)

        if self.remote is not None:
            self.remote.upload_object(basename, filename)

        self.samples_per_shard.append(len(self.new_samples))
        self.bytes_per_shard.append(self.new_shard_size)
        self.new_samples = []
        self.new_shard_size = 0

    def _write_compression_scheme(self) -> None:
        """Save dataset compression metadata"""
        assert self.compression_scheme is not None, 'compression scheme should be set if writing this file'
        if self.new_samples:
            raise RuntimeError('Attempted to write compression metadata file while samples are still being processed.')
        basename = get_compression_scheme_basename()
        filename = os.path.join(self.dirname, basename)
        with open(filename, 'x') as out:
            out.write(self.compression_scheme)
        if self.remote is not None:
            self.remote.upload_object(basename, filename)

    def _write_index(self) -> None:
        """Save dataset index file."""
        if self.new_samples:
            raise RuntimeError('Attempted to write index file while samples are still being processed.')
        basename = get_index_basename(self.compression_scheme)
        filename = os.path.join(self.dirname, basename)
        samples_per_shard = np.array(self.samples_per_shard, np.int64)
        bytes_per_shard = np.array(self.bytes_per_shard, np.int64)
        bytes_per_sample = np.array(self.bytes_per_sample, np.int64)
        index = StreamingDatasetIndex(samples_per_shard, bytes_per_shard, bytes_per_sample, self.fields)
        with self._create_binary_file(filename) as out:
            index.dump(out)
        if self.remote is not None:
            self.remote.upload_object(basename, filename)

    def write_sample(self, sample: Dict[str, bytes]) -> None:
        """Add a sample to the dataset.

        Args:
            sample (Dict[str, bytes]): The new sample, whose keys must contain the fields to save (others ignored).
        """
        data = sample_dict_to_bytes(sample, self.fields)
        if self.shard_size_limit <= self.new_shard_size + len(data):
            self._flush_shard()
        self.bytes_per_sample.append(len(data))
        self.new_samples.append(data)
        self.new_shard_size += len(data)

    def write_samples(self,
                      samples: Iterable[Dict[str, bytes]],
                      use_tqdm: bool = True,
                      total: Optional[int] = None) -> None:
        """Add the samples from the given iterable to the dataset.

        Args:
            samples (Iterable[Dict[str, bytes]]): The new samples.
            use_tqdm (bool): Whether to display a progress bar.  Default: ``True``.
            total (int, optional): Total samples for the progress bar (for when samples is a generator).
        """
        if use_tqdm:
            samples = tqdm(samples, leave=False, total=total)
        for s in samples:
            self.write_sample(s)

    def finish(self) -> None:
        """Complete writing the dataset by flushing last samples to a last shard, then write an index file."""
        if self.new_samples:
            self._flush_shard()
        if self.compression_scheme is not None:
            self._write_compression_scheme()
        self._write_index()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.finish()


def _parse_remote(remote: Union[ObjectStore, str, None], timeout: Optional[float]) -> Optional[ObjectStore]:
    if isinstance(remote, str):
        return get_object_store(remote, timeout)
    elif isinstance(remote, ObjectStore):
        return remote
    elif remote is None:
        return None
    else:
        raise ValueError('Bad argument for remote')


def get_object_store(remote: str, timeout: Optional[float]) -> ObjectStore:
    """Use the correct download handler to download the file

    Args:
        remote (Optional[str]): Remote path (local filesystem).
        timeout (float): How long to wait for file to download before raising an exception.
    """
    if remote.startswith('s3://'):
        return get_s3_object_store(remote, timeout)
    elif remote.startswith('sftp://'):
        return get_sftp_object_store(remote)
    else:
        raise ValueError('unsupported upload scheme')


def get_s3_object_store(remote: str, timeout: Optional[float]) -> S3ObjectStore:
    if timeout is None:
        raise ValueError('Must specify timeout for s3 bucket')
    obj = urllib3.parse.urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")
    client_config = {'read_timeout': timeout}
    bucket = obj.netloc
    object_store = S3ObjectStore(bucket=bucket, client_config=client_config)
    return object_store


def get_sftp_object_store(remote: str) -> SFTPObjectStore:
    url = urllib.parse.urlsplit(remote)
    # Parse URL
    if url.scheme.lower() != 'sftp':
        raise ValueError('If specifying a URI, only the sftp scheme is supported.')
    if not url.hostname:
        raise ValueError('If specifying a URI, the URI must include the hostname.')
    if url.query or url.fragment:
        raise ValueError('Query and fragment parameters are not supported as part of a URI.')
    hostname = url.hostname
    port = url.port
    username = url.username
    password = url.password

    # Get SSH key file if specified
    key_filename = os.environ.get('COMPOSER_SFTP_KEY_FILE', None)
    known_hosts_filename = os.environ.get('COMPOSER_SFTP_KNOWN_HOSTS_FILE', None)

    # Default port
    port = port if port else 22

    object_store = SFTPObjectStore(
        host=hostname,
        port=port,
        username=username,
        password=password,
        known_hosts_filename=known_hosts_filename,
        key_filename=key_filename,
    )
    return object_store
