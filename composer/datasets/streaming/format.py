# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`StreamingDatsetIndex` format that defines shard/sample metadata for :class:`StreamingDataset`."""

from gzip import GzipFile
from io import BufferedIOBase, BufferedReader, BufferedWriter, BytesIO
from os.path import splitext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'get_index_basename',
    'get_shard_basename',
    'sample_dict_to_bytes',
    'bytes_to_sample_dict',
    'StreamingDatasetIndex',
]


def split_compression_suffix(local_path: str) -> Tuple[str, Optional[str]]:
    """Splits the compression suffix from a path

    Args:
        local_path (str): path to a (potentially) compressed file

    Returns:
        Tuple[str, str]: tuple containing decompressed filename and compression suffix, if one exists
    """
    decompressed_path, ext = splitext(local_path)
    if ext in ['.mds', '.txt', '.old']:
        return local_path, None

    return decompressed_path, ext[1:]


def get_compression_scheme_basename() -> str:
    """Get the basename for a streaming dataset index.

    Returns:
        str: Basename of file.
    """
    return 'compression.txt'


def get_index_basename(compression_name: Optional[str] = None) -> str:
    """Get the basename for a streaming dataset index.

    Args:
        compression_name (Optional[str]): compression extension of index file

    Returns:
        str: Basename of file.
    """
    compression_name = '.' + compression_name if compression_name is not None else ''
    return f'index.mds{compression_name}'


def get_shard_basename(shard: int, compression_name: Optional[str] = None) -> str:
    """Get the basename for a streaming dataset shard.

    Args:
        shard (int): Shard index.
        compression_name (Optional[str]): compression extension of shard file

    Returns:
        str: Basename of file.
        compression_name (Optional[str]): the compression scheme
    """
    compression_name = '.' + compression_name if compression_name is not None else ''
    return f'{shard:06}.mds{compression_name}'


def sample_dict_to_bytes(obj: Dict[str, bytes], keys: List[str]) -> bytes:
    """Dump a sample dict to bytes, given field names.

    Args:
        obj (Dict[str, bytes]): The sample dict to encode.
        keys (list of str): The field names.

    Returns:
        bytes: The encoded sample bytes.
    """
    values = []
    for key in keys:
        value = obj[key]
        values.append(value)
    sizes = list(map(len, values))
    sizes = np.array(sizes, np.int64)
    return sizes.tobytes() + b''.join(values)


def bytes_to_sample_dict(data: bytes, keys: List[str]) -> Dict[str, bytes]:
    """Load a sample dict from bytes and field names.

    Args:
        data (bytes): The encoded sample data.
        keys (List[str]): The field names. Must be in the same order as the ``keys`` used when calling
            :func:`.sample_dict_to_bytes`.

    Returns:
        Dict[str, bytes]: The decoded sample dict.
    """
    num_values = len(keys)
    sizes = np.frombuffer(data[:num_values * np.int64().nbytes], np.int64)
    ends = num_values * np.int64().nbytes + sizes.cumsum()
    begins = ends - sizes
    values = []
    for begin, end in zip(begins, ends):
        value = data[begin:end]
        values.append(value)
    return dict(zip(keys, values))


def read_array(fp: Union[BufferedIOBase, GzipFile], count: int, dtype: type) -> np.ndarray:
    """Load the count items from the file handle, advancing its position.

    Args:
        fp (BufferedIOBase): File handle.
        count (int): Number of items to read.
        dtype (type): Item datatype.

    Returns:
        np.ndarray: The read array.
    """
    num_bytes = count * dtype().nbytes
    data = fp.read(num_bytes)
    return np.frombuffer(data, dtype)


class StreamingDatasetIndex(object):
    """Streaming Dataset index file, containing all the info about shards and samples.

    The shards are binary buffers with samples concatenated together. All the offset info across the whole dataset is
    contained in the index file. Workers read this file to calculate how much of which shards their slice is.

    Each sample is a dict of str to bytes. All samples must contain the same dict keys (fields). These strings are
    stored in the index file for efficiency.

    Args:
        samples_per_shard (NDArray[np.int64]): Number of samples of each shard.
        bytes_per_shard (NDArray[np.int64]): Size in bytes of each shard.
        bytes_per_sample (NDArray[np.int64]): Size in bytes of each sample across all shards.
        fields (List[str]): The names of the samples' fields in order.
    """

    def __init__(self,
                 samples_per_shard: NDArray[np.int64],
                 bytes_per_shard: NDArray[np.int64],
                 bytes_per_sample: NDArray[np.int64],
                 fields: List[str],
                 shuffle_indices: Optional[NDArray[np.int64]] = None) -> None:
        self.samples_per_shard = samples_per_shard
        self.bytes_per_shard = bytes_per_shard
        self.bytes_per_sample = bytes_per_sample
        self.fields = fields

        # Totals.
        self.num_shards = len(samples_per_shard)
        self.total_bytes = sum(bytes_per_shard)
        self.total_samples = len(bytes_per_sample)
        self.num_fields = len(fields)

        # Shard -> sample range.
        self.shard_ends = self.samples_per_shard.cumsum()
        self.shard_begins = self.shard_ends - self.samples_per_shard

        # Sample -> shard, byte offset within shard.
        self.shuffle_indices = shuffle_indices if shuffle_indices is not None else np.arange(self.num_shards)
        self.sample_shards, self.sample_id_shards, self.sample_shard_offsets, self.shard_samples = self._locate_samples(
        )

    def relocate_samples(self, shuffle_indices: NDArray[np.int64]) -> None:
        self.shuffle_indices = shuffle_indices
        self.sample_shards, self.sample_id_shards, self.sample_shard_offsets, self.shard_samples = self._locate_samples(
        )

    @classmethod
    def loads(cls, data: bytes):
        """Load a StreamingDatasetIndex from raw bytes.

        Args:
            data (bytes): The serialized form.

        Returns:
            cls: The loaded object.
        """
        fp = BytesIO(data)
        return cls.load(fp)

    @classmethod
    def load(cls, fp: Union[BufferedReader, BytesIO, GzipFile], shuffle_indices: Optional[NDArray[np.int64]] = None):
        """Load a StreamingDatasetIndex from a file handle.

        Args:
            fp (file): The file to read.

        Returns:
            cls: The loaded object.
        """
        magic, version, num_shards = read_array(fp, 3, np.uint32)
        assert magic == 0xDA7AD06E
        assert version == 1
        total_samples, total_bytes = read_array(fp, 2, np.int64)
        del total_bytes
        samples_per_shard = read_array(fp, num_shards, np.int64)
        bytes_per_shard = read_array(fp, num_shards, np.int64)
        bps_format, = read_array(fp, 1, np.int32)
        if not bps_format:
            sample_bytes, = read_array(fp, 1, np.int64)
            bytes_per_sample = np.full(total_samples, sample_bytes)
        elif bps_format == 1:
            bytes_per_sample = read_array(fp, total_samples, np.int8)
        elif bps_format == 2:
            bytes_per_sample = read_array(fp, total_samples, np.int16)
        elif bps_format == 4:
            bytes_per_sample = read_array(fp, total_samples, np.int32)
        elif bps_format == 8:
            bytes_per_sample = read_array(fp, total_samples, np.int64)
        else:
            assert False
        bytes_per_sample = bytes_per_sample.astype(np.int64)
        num_fields, = read_array(fp, 1, np.int32)
        bytes_per_field = read_array(fp, num_fields, np.int32)
        fields = [fp.read(size).decode('utf-8') for size in bytes_per_field]
        return cls(samples_per_shard, bytes_per_shard, bytes_per_sample, fields, shuffle_indices)

    def dumps(self) -> bytes:
        """Dump a StreamingDatasetIndex to raw bytes.

        Returns:
            bytes: The serialized form.
        """
        magic = 0xDA7AD06E
        version = 1
        header = np.array([magic, version, self.num_shards], np.uint32)
        totals = np.array([self.total_samples, self.total_bytes], np.int64)
        if not len(self.bytes_per_sample):
            bps_format = 1
            bps = self.bytes_per_sample.astype(np.int8)
        elif len(set(self.bytes_per_sample)) == 1:
            bps_format = 0
            bps = np.int64(self.bytes_per_sample[0])
        else:
            max_bps = self.bytes_per_sample.max()
            if max_bps < 256:
                bps_format = 1
                bps = self.bytes_per_sample.astype(np.int8)
            elif max_bps < (1 << 16):
                bps_format = 2
                bps = self.bytes_per_sample.astype(np.int16)
            elif max_bps < (1 << 32):
                bps_format = 4
                bps = self.bytes_per_sample.astype(np.int32)
            else:
                bps_format = 8
                bps = self.bytes_per_sample
        bps_format = np.int32(bps_format)
        num_fields = np.int32(len(self.fields))
        bytes_per_field = np.array([len(field.encode('utf-8')) for field in self.fields], np.int32)
        arrays = (header, totals, self.samples_per_shard, self.bytes_per_shard, bps_format, bps, num_fields,
                  bytes_per_field)
        array_bytes = b''.join([arr.tobytes() for arr in arrays])
        field_bytes = b''.join([field.encode('utf-8') for field in self.fields])
        return array_bytes + field_bytes

    def dump(self, fp: Union[BufferedWriter, GzipFile]) -> None:
        """Dump a StreamingDatasetIndex to the file.

        Args:
            fp (file): The file to write.
        """
        data = self.dumps()
        fp.write(data)

    def _locate_samples(self) -> Tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Precompute the shard and byte offset within the shard of every sample.

        Returns:
            sample_shards (NDArray[np.int64]): Shard per sample.
            sample_shard_offsets (NDArray[np.int64]): Intra-shard byte offset per sample.
        """
        samples_per_shard_shuffled = self.samples_per_shard[self.shuffle_indices]
        indices = self.shuffle_indices

        shard_ends = self.bytes_per_shard.cumsum()
        shard_begins = shard_ends - self.bytes_per_shard

        sample_shard_begins = []
        sample_id_shards = []
        sample_shards = []
        shard_samples = []
        for shard, (num_samples, shard_begin) in enumerate(zip(self.samples_per_shard, shard_begins)):
            sample_id_shards += [shard] * num_samples
            sample_shard_begins += [shard_begin] * num_samples

        for (shard, num_samples) in zip(indices, samples_per_shard_shuffled):
            sample_shards += [shard] * num_samples
            shard_samples.append(num_samples)

        sample_shard_begins = np.array(sample_shard_begins, np.int64)
        sample_id_shards = np.array(sample_id_shards, np.int64)
        sample_shards = np.array(sample_shards, np.int64)
        shard_samples = np.array(shard_samples, np.int64)

        sample_ends = self.bytes_per_sample.astype(np.int64).cumsum()
        sample_begins = sample_ends - self.bytes_per_sample
        sample_shard_offsets = sample_begins - sample_shard_begins
        return sample_shards, sample_id_shards, sample_shard_offsets, shard_samples
