# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`StreamingDatsetIndex` format that defines shard/sample metadata for :class:`StreamingDataset`.
"""

import math
from io import BufferedIOBase, BufferedReader, BufferedWriter, BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from composer.datasets.streaming.world import World

__all__ = [
    "get_index_basename",
    "get_shard_basename",
    "sample_dict_to_bytes",
    "bytes_to_sample_dict",
    "StreamingDatasetIndex",
]


def get_index_basename() -> str:
    """Get the basename for a streaming dataset index.

    Returns:
        str: Basename of file.
    """
    return 'index.mds'


def get_shard_basename(shard: int) -> str:
    """Get the basename for a streaming dataset shard.

    Args:
        shard (int): Shard index.

    Returns:
        str: Basename of file.
    """
    return f'{shard:06}.mds'


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
        keys (List[str]): The field names. Must be in the same order as the ``keys`` used when calling :func:`.sample_dict_to_bytes`.

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


def read_array(fp: BufferedIOBase, count: int, dtype: type) -> np.ndarray:
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

    The shards are binary buffers with samples concatenated together. All the
    offset info across the whole dataset is contained in the index file. Workers
    read this file to calculate how much of which shards their slice is.

    Each sample is a dict of str to bytes. All samples must contain the same
    dict keys (fields). These strings are stored in the index file for
    efficiency.

    Args:
        samples_per_shard (NDArray[np.int64]): Number of samples of each shard.
        bytes_per_shard (NDArray[np.int64]): Size in bytes of each shard.
        bytes_per_sample (NDArray[np.int64]): Size in bytes of each sample across all shards.
        fields (List[str]): The names of the samples' fields in order.
    """

    def __init__(self, samples_per_shard: NDArray[np.int64], bytes_per_shard: NDArray[np.int64],
                 bytes_per_sample: NDArray[np.int64], fields: List[str]) -> None:

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
        self.sample_shards, self.sample_shard_offsets = self._locate_samples()

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
    def load(cls, fp: Union[BufferedReader, BytesIO]):
        """Load a StreamingDatasetIndex from a file handle.

        Args:
            fp (file): The file to read.

        Returns:
            cls: The loaded object.
        """

        dtype = np.int64
        total_samples, total_bytes, num_shards, num_fields = read_array(fp, 4, dtype)
        del total_bytes
        samples_per_shard = read_array(fp, num_shards, dtype)
        bytes_per_shard = read_array(fp, num_shards, dtype)
        bytes_per_sample = read_array(fp, total_samples, dtype)
        bytes_per_field = read_array(fp, num_fields, dtype)
        fields = [fp.read(size).decode('utf-8') for size in bytes_per_field]
        return cls(samples_per_shard, bytes_per_shard, bytes_per_sample, fields)

    def dumps(self) -> bytes:
        """Dump a StreamingDatasetIndex to raw bytes.

        Returns:
            bytes: The serialized form.
        """
        header = np.array([self.total_samples, self.total_bytes, self.num_shards, self.num_fields], np.int64)
        bytes_per_field = np.array(list(map(len, self.fields)), np.int64)
        arrays = header, self.samples_per_shard, self.bytes_per_shard, self.bytes_per_sample, bytes_per_field
        arrays = np.concatenate(arrays, dtype=np.int64).tobytes()
        fields = b''.join(map(lambda s: s.encode('utf-8'), self.fields))
        return arrays + fields

    def dump(self, fp: BufferedWriter) -> None:
        """Dump a StreamingDatasetIndex to the file.

        Args:
            fp (file): The file to write.
        """
        data = self.dumps()
        fp.write(data)

    def _locate_samples(self) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Precompute the shard and byte offset within the shard of every sample.

        Returns:
            sample_shards (NDArray[np.int64]): Shard per sample.
            sample_shard_offsets (NDArray[np.int64]): Intra-shard byte offset per sample.
        """
        shard_ends = self.bytes_per_shard.cumsum()
        shard_begins = shard_ends - self.bytes_per_shard

        sample_shard_begins = []
        sample_shards = []
        for shard, (num_samples, shard_begin) in enumerate(zip(self.samples_per_shard, shard_begins)):
            sample_shard_begins += [shard_begin] * num_samples
            sample_shards += [shard] * num_samples
        sample_shard_begins = np.array(sample_shard_begins, np.int64)
        sample_shards = np.array(sample_shards, np.int64)

        sample_ends = self.bytes_per_sample.cumsum()
        sample_begins = sample_ends - self.bytes_per_sample
        sample_shard_offsets = sample_begins - sample_shard_begins
        return sample_shards, sample_shard_offsets

    def get_partition(self, world: World, batch_size: Optional[int] = None) -> Tuple[List[int], int, int]:
        """Get the shards and sample range of a given partition of the dataset.

        Args:
            world (World): Context about workers, devices, and nodes.
            batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader.
                                    Worker indices will be constructed so that there is at most 1 incomplete batch at the end of each epoch.
                                    E.g. if the DataLoader is reading over (samples=[0, 1, 2, 3, 4, 5, 6, 7], num_workers=3, batch_size=2, drop_last=True)
                                    but `batch_size` is not hinted to the StreamingDataset ahead of time
                                    then the samples will by default be assigned like: w0: [0, 1, 2], w1: [3, 4, 5], w2: [6, 7]
                                    and will be read as batches: [0, 1], [3, 4], [6, 7] (with batches [2] and [5] dropped as incomplete)
                                    but this is suboptimal because we could have dropped no samples.
                                    So when `batch_size` is provided as a hint, we assign samples like this: w0: [0, 1, 2, 3], w1: [4, 5], w2: [6, 7]
                                    which will be read as batches: [0, 1], [4, 5], [6, 7], [2, 3]

        Returns:
            shards (Sequence[int]): The shards that this partition overlaps.
            min_id (int): The lowest sample ID of this partition.
            max_id (int): The highest sample ID of this partition.
        """

        global_device = world.global_device
        global_num_devices = world.global_num_devices
        device_worker = world.device_worker
        device_num_workers = world.device_num_workers

        # Splits a range (start, start+total) into num_parts such that:
        # each part spans a continguous range [part_min_id, part_max_id]
        # each part_i starts immediately from where the previous part_[i-1] stopped
        # all parts have the same number of items,
        # except the first K parts may have exactly 1 more item
        def _get_min_max_size(start: int, total: int, part: int, num_parts: int):
            sizes = [math.ceil((total - p) / num_parts) for p in range(num_parts)]
            min_ids = np.cumsum([0] + sizes)
            part_min_id = start + min_ids[part]
            part_max_id = start + min_ids[part + 1] - 1
            part_size = sizes[part]
            return part_min_id, part_max_id, part_size

        device_min_id, _, device_samples = _get_min_max_size(0, self.total_samples, global_device, global_num_devices)

        # Some devices may have 1 fewer sample, so repeat some samples at boundaries
        expected_device_samples = math.ceil(self.total_samples / global_num_devices)
        if device_samples < expected_device_samples:
            if device_samples != expected_device_samples - 1:
                raise RuntimeError("Found device partition with incorrect # samples")
            device_min_id -= 1
            device_samples += 1

        if not batch_size:
            worker_min_id, worker_max_id, _ = _get_min_max_size(device_min_id, device_samples, device_worker,
                                                                device_num_workers)
        else:
            device_batches = math.ceil(device_samples / batch_size)
            samples_missing = device_batches * batch_size - device_samples

            # Determine which batches this worker is responsible for
            worker_min_batch_id, worker_max_batch_id, _ = _get_min_max_size(0, device_batches, device_worker,
                                                                            device_num_workers)

            # The last device_worker to be read from will be the one with the incomplete batch.
            # This is done to match PyTorch DataLoader's round-robin scheduling of workers
            # All device_workers must be careful to account for the missing samples offset by the incomplete batch
            incomplete_device_worker = (device_batches + device_num_workers - 1) % device_num_workers
            min_id_offset = 0 if device_worker <= incomplete_device_worker else samples_missing
            max_id_offset = 0 if device_worker < incomplete_device_worker else samples_missing

            worker_min_id = device_min_id + worker_min_batch_id * batch_size - min_id_offset
            worker_max_id = device_min_id + (worker_max_batch_id + 1) * batch_size - max_id_offset - 1

        min_shard = self.sample_shards[worker_min_id]
        max_shard = self.sample_shards[worker_max_id]
        shards = list(range(min_shard, max_shard + 1))
        return shards, worker_min_id, worker_max_id
