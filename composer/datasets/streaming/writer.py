# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

""":class:`StreamingDatasetWriter` is used to convert a list of samples into binary `.mds` files that can be read as a :class:`StreamingDataset`.
"""

import os
from types import TracebackType
from typing import Dict, Iterable, List, Optional, Type

import numpy as np
from tqdm import tqdm

from composer.datasets.streaming.format import (StreamingDatasetIndex, get_index_basename, get_shard_basename,
                                                sample_dict_to_bytes)

__all__ = ["StreamingDatasetWriter"]


class StreamingDatasetWriter(object):
    """
    Used for writing a :class:`StreamingDataset` from a list of samples.

    Samples are expected to be of type: `Dict[str, bytes]`.

    Given each sample, :class:`StreamingDatasetWriter` only writes out the values for a subset of keys (`fields`) that are globally shared across the dataset.

    :class:`StreamingDatasetWriter` automatically shards the dataset such that each shard is of size <= `shard_size_limit` bytes.


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
        shard_size_limit (int): Maximum shard size in bytes. Default: `1 << 24`.
    """

    def __init__(self, dirname: str, fields: List[str], shard_size_limit: int = 1 << 24) -> None:
        if len(fields) != len(set(fields)):
            raise ValueError(f"fields={fields} must be unique.")
        if shard_size_limit <= 0:
            raise ValueError(f"shard_size_limit={shard_size_limit} must be positive.")

        self.dirname = dirname
        self.fields = fields
        self.shard_size_limit = shard_size_limit

        # Stats about shards written so far.
        self.samples_per_shard = []
        self.bytes_per_shard = []
        self.bytes_per_sample = []

        # Data of the shard in progress.
        self.new_samples = []
        self.new_shard_size = 0

    def _flush_shard(self) -> None:
        """Flush cached samples to a new dataset shard."""
        os.makedirs(self.dirname, exist_ok=True)
        shard = len(self.samples_per_shard)
        basename = get_shard_basename(shard)
        filename = os.path.join(self.dirname, basename)
        with open(filename, 'xb') as out:
            for data in self.new_samples:
                out.write(data)
        self.samples_per_shard.append(len(self.new_samples))
        self.bytes_per_shard.append(self.new_shard_size)
        self.new_samples = []
        self.new_shard_size = 0

    def _write_index(self) -> None:
        """Save dataset index file."""
        if self.new_samples:
            raise RuntimeError("Attempted to write index file while samples are still being processed.")
        filename = os.path.join(self.dirname, get_index_basename())
        ndarray_samples_per_shard = np.asarray(self.samples_per_shard, np.int64)
        ndarray_bytes_per_shard = np.asarray(self.bytes_per_shard, np.int64)
        ndarray_bytes_per_sample = np.asarray(self.bytes_per_sample, np.int64)
        index = StreamingDatasetIndex(ndarray_samples_per_shard, ndarray_bytes_per_shard, ndarray_bytes_per_sample,
                                      self.fields)
        with open(filename, 'xb') as out:
            index.dump(out)

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
        self._write_index()

    def __enter__(self):
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.finish()
