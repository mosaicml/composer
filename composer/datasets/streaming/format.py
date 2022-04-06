from io import BufferedReader, BufferedWriter
from typing import Dict, List, Sequence, Tuple

import numpy as np


def get_index_basename() -> str:
    """Get the basename for a streaming dataset index.

    Returns:
        str: Basename of file.
    """
    return 'index.mds'


def get_shard_basename(shard: int) -> str:
    """Get the basename for a streaming dataset shard.

    Args:
        shard (int): Which shard.

    Returns:
        str: Basename of file.
    """
    return '%05d.mds' % shard


def sample_dict_to_bytes(obj: Dict[str, bytes], keys: List[str]) -> bytes:
    """Dump a sample dict to bytes, given field names.

    Args:
        obj (dict): The sample dict to encode.
        keys (list of str): The field names.

    Returns:
        bytes: The encoded sample bytes.
    """
    values = []
    for key in keys:
        value = obj[key]
        values.append(value)
    sizes = list(map(len, values))
    sizes = np.array(sizes, np.int32)
    return sizes.tobytes() + b''.join(values)


def bytes_to_sample_dict(data: bytes, keys: List[str]) -> Dict[str, bytes]:
    """Load a sample dict from bytes and field names.

    Args:
        data (bytes): The encoded sample data.
        keys (list of str): The field names.

    Returns:
        dict: The decoded sample dict.
    """
    num_values = len(keys)
    sizes = np.frombuffer(data[:num_values * 4], np.int32)
    ends = num_values * 4 + sizes.cumsum(0)
    begins = ends - sizes
    values = []
    for begin, end in zip(begins, ends):
        value = data[begin:end]
        values.append(value)
    return dict(zip(keys, values))


class StreamingDatasetIndex(object):
    """Streaming dataset index file, giving all the information about shards.

    The shards are just dumb buffers with samples catted together. All the offset info across the whole dataset is
    contained in the index file. Workers read this file to calculate how much of which shards their slice is.

    Each sample is a dict of str to bytes. All samples must contain the same dict keys (fields). These strings are
    stored in the index file for efficiency.

    Format:
    - Num shards
    - Total samples
    - Total bytes
    - Num fields
    - Samples per shard
    - Bytes per shard
    - Bytes per sample
    - Bytes per field
    - Fields
    """

    def __init__(self, samples_per_shard: Sequence[int], bytes_per_shard: Sequence[int],
                 bytes_per_sample: Sequence[int], fields: List[str]) -> None:
        """Initialize with sample stats.

        Args:
            samples_per_shard (sequence of int): Number of samples of each shard.
            bytes_per_shard (sequence of int): Size in bytes of each shard.
            bytes_per_sample (sequence of int): Size in bytes of each sample across all shards.
            fields (list of str): The names of the sample's fields in order.
        """
        self.samples_per_shard = np.asarray(samples_per_shard, np.int32)
        self.bytes_per_shard = np.asarray(bytes_per_shard, np.int32)
        self.bytes_per_sample = np.asarray(bytes_per_sample, np.int32)
        self.fields = fields

        self.num_shards = len(samples_per_shard)
        self.total_samples = len(bytes_per_sample)
        self.total_bytes = sum(bytes_per_shard)
        self.num_fields = len(fields)

    @classmethod
    def loads(cls, data: bytes):
        """Load a StreamingDatasetIndex from raw bytes.

        Args:
            data (bytes): The serialized form.

        Returns:
            cls: The loaded object.
        """
        begin = 0
        end = begin + 4 * 4
        num_shards, total_samples, total_bytes, num_fields = np.frombuffer(data[begin:end], np.int32)

        begin = end
        end = begin + num_shards * 4
        samples_per_shard = np.frombuffer(data[begin:end], np.int32)

        begin = end
        end = begin + num_shards * 4
        bytes_per_shard = np.frombuffer(data[begin:end], np.int32)
        assert sum(bytes_per_shard) == total_bytes

        begin = end
        end = begin + total_samples * 4
        bytes_per_sample = np.frombuffer(data[begin:end], np.int32)

        begin = end
        end = begin + num_fields * 4
        bytes_per_field = np.frombuffer(data[begin:end], np.int32)

        fields = []
        for size in bytes_per_field:
            begin = end
            end = begin + size
            field = data[begin:end].decode('utf-8')
            fields.append(field)

        return cls(samples_per_shard, bytes_per_shard, bytes_per_sample, fields)

    @classmethod
    def load(cls, fp: BufferedReader):
        """Load a StreamingDatasetIndex from a file handle.

        Args:
            fp (file): The file to read.

        Returns:
            cls: The loaded object.
        """
        data = fp.read()
        return cls.loads(data)

    def dumps(self) -> bytes:
        """Dump a StreamingDatasetIndex to raw bytes.

        Returns:
            bytes: The serialized form.
        """
        counts = self.num_shards, self.total_samples, self.total_bytes, self.num_fields
        bytes_per_field = list(map(len, self.fields))
        arrays = counts, self.samples_per_shard, self.bytes_per_shard, self.bytes_per_sample, bytes_per_field
        ints = np.concatenate(arrays).astype(np.int32)
        byte_fields = list(map(lambda s: s.encode('utf-8'), self.fields))
        return ints.tobytes() + b''.join(byte_fields)

    def dump(self, fp: BufferedWriter) -> None:
        """Dump a StreamingDatasetIndex to the file.

        Args:
            fp (file): The file to write.
        """
        data = self.dumps()
        fp.write(data)

    def locate_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute the shard and byte offset within the shard of every sample.

        Returns:
            sample_shards (np.ndarray): Shard per sample.
            sample_shard_offsets (np.ndarray): Intra-shard byte offset per sample.
        """
        shard_ends = self.bytes_per_shard.cumsum(0)
        shard_begins = shard_ends - self.bytes_per_shard

        sample_shard_begins = []
        sample_shards = []
        for shard, (num_samples, shard_begin) in enumerate(zip(self.samples_per_shard, shard_begins)):
            sample_shard_begins += [shard_begin] * num_samples
            sample_shards += [shard] * num_samples
        sample_shard_begins = np.array(sample_shard_begins, np.int32)
        sample_shards = np.array(sample_shards, np.int32)

        sample_ends = self.bytes_per_sample.cumsum(0)
        sample_begins = sample_ends - self.bytes_per_sample
        sample_shard_offsets = sample_begins - sample_shard_begins
        return sample_shards, sample_shard_offsets

    def get_shard_sample_range(self, shard: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the dataset-wide sample range of a shard.

        Args:
            shard (int): Which shard.

        Returns:
            begin (np.ndarray): First sample ID of shard.
            end (np.ndarray): One past the last sample ID of the shard.
        """
        ends = self.samples_per_shard.cumsum(0)
        begins = ends - self.samples_per_shard
        return begins[shard], ends[shard]
