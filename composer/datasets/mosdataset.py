import numpy as np
import os
from typing import Dict, List, Optional, Tuple


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
    n_values = len(keys)
    sizes = np.frombuffer(data[:n_values * 4], np.int32)
    ends = n_values * 4 + sizes.cumsum(0)
    begins = ends - sizes
    values = []
    for begin, end in zip(begins, ends):
        value = data[begin:end]
        values.append(value)
    return dict(zip(keys, values))


class MosaicDatasetIndex(object):
    """Mosaic dataset index file, giving all the information about shards.

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

    def __init__(self, samples_per_shard: np.ndarray, bytes_per_shard: np.ndarray, bytes_per_sample: np.ndarray,
                 fields: List[str]) -> None:
        """Initialize with sample stats.

        Args:
            samples_per_shard (np.ndarray): Number of samples of each shard.
            bytes_per_shard (np.ndarray): Size in bytes of each shard.
            bytes_per_sample (np.ndarray): Size in bytes of each sample across all shards.
            fields (list of str): The names 
        """
        self.samples_per_shard = samples_per_shard
        self.bytes_per_shard = bytes_per_shard
        self.bytes_per_sample = bytes_per_sample
        self.fields = fields

        self.num_shards = len(samples_per_shard)
        self.total_samples = len(bytes_per_sample)
        self.total_bytes = sum(bytes_per_sample)
        self.num_fields = len(fields)

    @classmethod
    def loads(cls, data):
        """Load a MosaicDatasetIndex from raw bytes.

        Args:
            data (bytes): The serialized form.

        Returns:
            cls: The loaded object.
        """
        begin = 0
        end = begin + 4 * 4
        num_shards, num_samples, num_bytes, num_fields = np.frombuffer(data[begin:end], np.int32)

        begin = end
        end = begin + num_shards * 4
        samples_per_shard = np.frombuffer(data[begin:end], np.int32)

        begin = end
        end = begin + num_shards * 4
        bytes_per_shard = np.frombuffer(data[begin:end], np.int32)

        begin = end
        end = begin + num_samples * 4
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
    def load(cls, fp):
        """Load a MosaicDatasetIndex from a file handle.

        Args:
            fp (file): The file to read.

        Returns:
            cls: The loaded object.
        """
        data = fp.read()
        return cls.loads(data)

    def dumps(self) -> None:
        """Dump a MosaicDatasetIndex to raw bytes.

        Returns:
            bytes: The serialized form.
        """
        bytes_per_field = list(map(len, self.fields))
        ints = np.concatenate([[self.num_shards, self.total_samples, self.total_bytes, self.num_fields],
                               self.samples_per_shard, self.bytes_per_shard, self.bytes_per_sample,
                               bytes_per_field]).astype(np.int32)
        byte_fields = list(map(lambda s: s.encode('utf-8'), self.fields))
        return ints.tobytes() + b''.join(byte_fields)

    def dump(self, fp) -> None:
        """Dump a MosaicDatasetIndex to the file.

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
        for shard, (n_samples, shard_begin) in enumerate(zip(self.samples_per_shard, shard_begins)):
            sample_shard_begins += [shard_begin] * n_samples
            sample_shards += [shard] * n_samples
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
            begin: First sample ID of shard.
            end: One past the last sample ID of the shard.
        """
        ends = self.samples_per_shard.cumsum(0)
        begins = ends - self.samples_per_shard
        return begins[shard], ends[shard]


class MosaicDatasetWriter(object):
    """Writes MosaicDatasets."""

    def __init__(self, dirname: str, fields: List[str], shard_size_limit: int = 1 << 24) -> None:
        """Initialize with the given output dirname.

        Args:
            dirname (str): Directory to write shards to.
            fields: (list of str): The fields to save for each sample.
            shard_size_limit (int, optional): Maximum shard size in bytes. Default: 1 << 24.
        """
        assert len(fields) == len(set(fields))
        assert 1 <= shard_size_limit

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
        if not self.samples_per_shard:
            os.makedirs(self.dirname)
        shard_idx = len(self.samples_per_shard)
        filename = os.path.join(self.dirname, '%05d.mds' % shard_idx)
        with open(filename, 'wb') as out:
            for data in self.new_samples:
                out.write(data)
        self.samples_per_shard.append(len(self.new_samples))
        self.bytes_per_shard.append(self.new_shard_size)
        self.new_samples = []
        self.new_shard_size = 0

    def _write_index(self) -> None:
        """Save dataset index file."""
        assert not self.new_samples
        filename = os.path.join(self.dirname, 'index.mds')
        index = MosaicDatasetIndex(self.samples_per_shard, self.bytes_per_shard, self.bytes_per_sample, self.fields)
        with open(filename, 'wb') as out:
            index.dump(out)

    def write_sample(self, obj: Dict[str, bytes]) -> None:
        """Add a sample to the dataset.

        Args:
            obj (dict): The new sample, whose keys must contain the fields to save (others ignored).
        """
        data = sample_dict_to_bytes(obj, self.fields)
        if self.shard_size_limit <= self.new_shard_size + len(data):
            self._flush_shard()
        self.bytes_per_sample.append(len(data))
        self.new_samples.append(data)
        self.new_shard_size += len(data)

    def write_samples(self, objs: Iterable[Dict[str, bytes]], use_tqdm: bool = True,
                      total: Optional[int] = None) -> None:
        """Add the samples from the given iterable to the dataset.

        Args:
            objs (iterable of dict): The new samples.
            use_tqdm (bool): Whether to display a progress bar.
            total (int, optional): Total samples for the progress bar (for when objs is a generator).
        """
        if use_tqdm:
            objs = tqdm(objs, leave=False, total=total)
        for obj in objs:
            self.write_sample(obj)

    def finish(self) -> None:
        """Complete writing the dataset by flushing last samples to a last shard, then write an index file."""
        if self.new_samples:
            self._flush_shard()
        self._write_index()

    def __enter__(self):
        """Enter as contextmanager."""
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Exit as contextmanager."""
        self.finish()
