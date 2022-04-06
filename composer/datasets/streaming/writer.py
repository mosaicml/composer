import os
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from composer.datasets.streaming.format import (StreamingDatasetIndex, get_index_basename, get_shard_basename,
                                                sample_dict_to_bytes)


class StreamingDatasetWriter(object):
    """Writes StreamingDatasets."""

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
        assert not self.new_samples
        filename = os.path.join(self.dirname, get_index_basename())
        index = StreamingDatasetIndex(self.samples_per_shard, self.bytes_per_shard, self.bytes_per_sample, self.fields)
        with open(filename, 'xb') as out:
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

    def write_samples(self,
                      objs: Iterable[Dict[str, bytes]],
                      use_tqdm: bool = True,
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

    def __exit__(self, *args: List, **kwargs: Dict) -> None:
        """Exit as contextmanager."""
        self.finish()
