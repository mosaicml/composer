# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`StreamingDataset` class, used for building streaming iterable datasets.
"""

import math
import os
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset

from composer.datasets.streaming.download import download_or_wait
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict, get_index_basename,
                                                get_shard_basename)
from composer.datasets.streaming.world import get_world
from composer.utils import dist

__all__ = ['StreamingDataset']


class StreamingDataset(IterableDataset):
    """A sharded, streaming, iterable dataset.

    :class:`StreamingDataset` reads samples from binary `.mds` files that were written out by :class:`StreamingDatasetWriter`.

    It currently supports downloading data from either remote paths (S3, SFTP) or local filepaths.

    It supports multi-gpu + multi-node training, and has smart local cacheing to minimize network bandwidth.

    It also provides best-effort shuffling to preserve randomness when ``shuffle=True``.

    Args:
        remote (str): Download shards from this remote path or directory.
        local (str): Download shards to this local directory for for caching.
        shuffle (bool): Whether to shuffle the samples.  Note that if `shuffle=False`, the sample order is deterministic but dependent on the DataLoader's `num_workers`.
        decoders (Dict[str, Callable[bytes, Any]]]): For each sample field you wish to read, you must provide a decoder to convert the raw bytes to an object.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 60 sec.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default: ``None``.
                                    Worker indices will be constructed so that there is at most 1 incomplete batch at the end of each epoch.
                                    E.g. if the DataLoader is reading over (samples=[0, 1, 2, 3, 4, 5, 6, 7], num_workers=3, batch_size=2, drop_last=True)
                                    but `batch_size` is not hinted to the StreamingDataset ahead of time
                                    then the samples will by default be assigned like: w0: [0, 1, 2], w1: [3, 4, 5], w2: [6, 7]
                                    and will be read as batches: [0, 1], [3, 4], [6, 7] (with batches [2] and [5] dropped as incomplete)
                                    but this is suboptimal because we could have dropped no samples.
                                    So when `batch_size` is provided as a hint, we assign samples like this: w0: [0, 1, 2, 3], w1: [4, 5], w2: [6, 7]
                                    which will be read as batches: [0, 1], [4, 5], [6, 7], [2, 3]


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
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 decoders: Dict[str, Callable[[bytes], Any]],
                 max_retries: int = 2,
                 timeout: float = 60,
                 batch_size: Optional[int] = None) -> None:

        self.remote = remote
        self.local = local
        self.shuffle = shuffle
        self.decoders = decoders
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size

        # Load the index file containing the shard metadata
        # This file contains the shard and offset in bytes of each sample (for direct access).
        # Only local device 0 on each node downloads the index. All other devices wait.
        index_basename = get_index_basename()
        index_local = self._download_file(index_basename, wait=(dist.get_local_rank() != 0))
        with open(index_local, 'rb') as fp:
            self.index = StreamingDatasetIndex.load(fp)

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock: Lock
        self._next_epoch = 0
        self._epoch_to_todo_ids = {}
        self._downloaded_ids = []
        self._is_downloaded = False

    def _download_file(self, basename: str, wait=False) -> str:
        """Safely download a file from remote to local cache.

        Args:
            basename (str): Basename of file to download.
            wait (bool): Whether to wait for another worker to download the file.

        Returns:
            str: Local cache filename.
        """
        remote = os.path.join(self.remote, basename)
        local = os.path.join(self.local, basename)
        download_or_wait(remote=remote, local=local, wait=wait, max_retries=self.max_retries, timeout=self.timeout)
        return local

    def _insert_shard_samples(self, shard: int, part_min_id: int, part_max_id: int) -> None:
        """Load the given locally cached shard into the dataset.

        Every time you call __iter__ on this dataset, it registers the list of
        samples you have left, which will not be the full epoch if the dataset
        isn't finished loaded when you start training.

        Calls to _insert_shard_samples during training modify the samples remaining on
        these iterations on the fly to insert these new samples and then re-sort,
        making the shuffle as perfect as was possible.

        This operation takes the lock, so batch your _insert_shard_samples calls where
        possible.

        Args:
            shard (int): Shard to load.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.
        """
        # Get all samples from the given shards that fall within our partition.
        shard_min_id = self.index.shard_begins[shard]
        shard_max_id = self.index.shard_ends[shard] - 1
        min_id = max(part_min_id, shard_min_id)
        max_id = min(part_max_id, shard_max_id)
        new_ids = list(range(min_id, max_id + 1))

        with self._lock:
            # Extend and optionally reshuffle the remaining samples of any
            # epochs we have in progress.
            if self.shuffle:
                if not self._is_downloaded:
                    self._downloaded_ids.extend(new_ids)
                    np.random.shuffle(self._downloaded_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)
                    np.random.shuffle(todo_ids)
            else:
                if not self._is_downloaded:
                    self._downloaded_ids.extend(new_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)

    def download(self) -> None:
        """Download and assimilate missing shards."""
        if not hasattr(self, '_lock'):
            self._lock = Lock()

        with self._lock:
            if self._is_downloaded:
                return

        # We find out num workers, and therefore num partitions, when __iter__ is called.
        # From the partition, derive our shard overlap range and exact sample range.
        world = get_world()
        part_shards, part_shards_to_download, part_min_id, part_max_id = self.index.get_partition(
            world, self.batch_size)

        if self.shuffle:
            # Always process first shard first because other workers may be waiting on it
            part_shards = np.array(part_shards)
            np.random.shuffle(part_shards[1:])

        for shard in part_shards:
            # If this worker is in charge of downloading the shard, download it.
            # Otherwise, wait until shard gets downloaded by another worker on this node
            # This produces deterministic sample order.
            basename = get_shard_basename(shard)
            self._download_file(basename, wait=(shard not in part_shards_to_download))
            self._insert_shard_samples(shard, part_min_id, part_max_id)

        with self._lock:
            self._is_downloaded = True

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Dataset length.
        """
        return math.ceil(self.index.total_samples / dist.get_world_size())

    def _unpack_sample(self, data: bytes) -> Dict[str, Any]:
        """Unpack a sample dict from raw bytes.

        First unpacks the str to raw bytes dict, then unpacks each field's raw bytes.

        Args:
            data (bytes): The packed bytes of the sample.

        Returns:
            Dict[str, Any]: The sample dict.
        """
        key_to_raw = bytes_to_sample_dict(data, self.index.fields)
        obj = {}
        for key, decode in self.decoders.items():
            raw_value = key_to_raw[key]
            decoded_value = decode(raw_value)
            obj[key] = decoded_value
        return obj

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index, assuming its shard is loaded.

        Do not call this directly unless the shard containing this idx has been loaded.
        Will crash otherwise.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: The sample.
        """
        shard = self.index.sample_shards[idx]
        offset = self.index.sample_shard_offsets[idx]
        size = self.index.bytes_per_sample[idx]

        basename = get_shard_basename(shard)
        shard_filename = os.path.join(self.local, basename)
        with open(shard_filename, 'rb', 0) as fp:
            fp.seek(offset)
            data = fp.read(size)

        return self._unpack_sample(data)

    def _make_new_growing_epoch(self) -> int:
        """Start a new growing epoch, in which we own the sample sequence because it grows.

        Returns:
            int: The epoch ID, an identifier which is given back to the caller.
        """
        with self._lock:
            epoch = self._next_epoch
            self._next_epoch += 1
            self._epoch_to_todo_ids[epoch] = list(self._downloaded_ids)
        return epoch

    def _next_id(self, epoch: int) -> Optional[int]:
        """Get next sample of the growing epoch given by epoch, or None if done.

        If we are currently out of samples but not finished downloading the
        shards, blocks until it has new samples.

        Args:
            epoch (int): The epoch, an identifier for this sequence of samples.

        Returns:
            int: ID of next sample.
        """
        while True:
            with self._lock:
                todo_ids = self._epoch_to_todo_ids[epoch]
                if todo_ids:
                    # Higher perf to pop last, but shuffle=False wants in-order traversal
                    if self.shuffle:
                        return todo_ids.pop(-1)
                    else:
                        return todo_ids.pop(0)
                elif self._is_downloaded:
                    del self._epoch_to_todo_ids[epoch]
                    return None
                else:
                    pass
            sleep(0.25)

    def _iter_ids(self) -> Iterator[int]:
        """Get an iterator over all our sample IDs.

        Returns:
            Iterator[int]: Each sample ID.
        """
        with self._lock:
            is_downloaded = self._is_downloaded

        if is_downloaded:
            ids = list(self._downloaded_ids)
            if self.shuffle:
                np.random.shuffle(ids)
            for idx in ids:
                yield idx
        else:
            epoch = self._make_new_growing_epoch()
            while True:
                idx = self._next_id(epoch)
                if idx is None:
                    break
                yield idx

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all the samples in our partition.

        If not all samples have been downloaded yet, iterates over what it has
        while inserting the remainder into the sequence behind the scenes as it
        progresses.

        Returns:
            Iterator[Any]: Each sample.
        """
        if not hasattr(self, '_lock'):
            self._lock = Lock()

        Thread(target=self.download, daemon=True).start()

        for idx in self._iter_ids():
            yield self[idx]
