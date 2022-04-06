from io import BufferedReader, BufferedWriter
from typing import Dict, Sequence, Tuple

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
    return f'{shard:05}.mds'


def sample_dict_to_bytes(obj: Dict[str, bytes], keys: Sequence[str]) -> bytes:
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


def bytes_to_sample_dict(data: bytes, keys: Sequence[str]) -> Dict[str, bytes]:
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
    """Streaming dataset index file, containing all the info about shards.

    The shards are binary buffers with samples concatenated together. All the
    offset info across the whole dataset is contained in the index file. Workers
    read this file to calculate how much of which shards their slice is.

    Each sample is a dict of str to bytes. All samples must contain the same
    dict keys (fields). These strings are stored in the index file for
    efficiency.
    """

    def __init__(self, samples_per_shard: Sequence[int], bytes_per_shard: Sequence[int],
                 bytes_per_sample: Sequence[int], fields: Sequence[str]) -> None:
        """Initialize with sample statistics.

        Args:
            samples_per_shard (Sequence[int]): Number of samples of each shard.
            bytes_per_shard (Sequence[int]): Size in bytes of each shard.
            bytes_per_sample (Sequence[int]): Size in bytes of each sample across all shards.
            fields (Sequence[str]): The names of the samples' fields in order.
        """
        self.samples_per_shard = np.asarray(samples_per_shard, np.int32)
        self.bytes_per_shard = np.asarray(bytes_per_shard, np.int32)
        self.bytes_per_sample = np.asarray(bytes_per_sample, np.int32)
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
        self.sample_shards, self.sample_shard_offsets = self.locate_samples()

    @classmethod
    def loads(cls, data: bytes):
        """Load a StreamingDatasetIndex from raw bytes.

        Args:
            data (bytes): The serialized form.

        Returns:
            cls: The loaded object.
        """
        begin = 0
        end = begin + 2 * 8
        total_samples, total_bytes = np.frombuffer(data[begin:end], np.int64)

        begin = end
        end = begin + 2 * 4
        num_shards, num_fields = np.frombuffer(data[begin:end], np.int32)

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
        x1 = np.array([self.total_samples, self.total_bytes], np.int64).tobytes()
        x2 = np.array([self.num_shards, self.num_fields], np.int32).tobytes()
        bytes_per_field = list(map(len, self.fields))
        arrays = self.samples_per_shard, self.bytes_per_shard, self.bytes_per_sample, bytes_per_field
        x3 = np.concatenate(arrays).astype(np.int32).tobytes()
        x4 = b''.join(map(lambda s: s.encode('utf-8'), self.fields))
        return x1 + x2 + x3 + x4

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
        shard_ends = self.bytes_per_shard.cumsum()
        shard_begins = shard_ends - self.bytes_per_shard

        sample_shard_begins = []
        sample_shards = []
        for shard, (num_samples, shard_begin) in enumerate(zip(self.samples_per_shard, shard_begins)):
            sample_shard_begins += [shard_begin] * num_samples
            sample_shards += [shard] * num_samples
        sample_shard_begins = np.array(sample_shard_begins, np.int32)
        sample_shards = np.array(sample_shards, np.int32)

        sample_ends = self.bytes_per_sample.cumsum()
        sample_begins = sample_ends - self.bytes_per_sample
        sample_shard_offsets = sample_begins - sample_shard_begins
        return sample_shards, sample_shard_offsets

    def get_partition_shards_and_samples(self, part_id: int, num_parts: int) -> Tuple[Sequence[int], int, int]:
        """Get the shards and sample range of a given partition of the dataset.

        Args:
            part_id (int): Which partition.
            num_parts (int): Out of how many partitions.

        Returns:
            shards (Sequence[int]): The shards that this partition overlaps.
            min_id (int): The lowest sample ID of this partition.
            max_id (int): The highest sample ID of this partition.
        """
        min_id = self.total_samples * part_id // num_parts
        max_id = self.total_samples * (part_id + 1) // num_parts - 1
        min_shard = self.sample_shards[min_id]
        max_shard = self.sample_shards[max_id]
        shards = list(range(min_shard, max_shard + 1))
        return shards, min_id, max_id
