# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`StreamingDataset` class, used for building streaming iterable datasets.
"""

import enum
import math
import os
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset

from composer.datasets.streaming.download import download_or_wait
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict,
                                                get_compression_scheme_basename, get_index_basename, get_shard_basename,
                                                split_compression_suffix)
from composer.datasets.streaming.shuffle import BlockCipherShuffler
from composer.datasets.streaming.world import get_world
from composer.utils import dist

__all__ = ['StreamingDataset']


class DatasetCompressionException(Exception):
    pass


class _DownloadStatus(enum.IntEnum):
    NOT_STARTED = 1
    IN_PROGRESS = 2
    DONE = 3
    FAILED = 4


class StreamingDataset(IterableDataset):
    """A sharded, streaming, iterable dataset.

    Features:

    * :class:`StreamingDataset` reads samples from binary ``.mds`` files that were written out by
      :class:`StreamingDatasetWriter`.
    * Supports downloading data from S3, SFTP, or local filesystem.
    * Supports multi-gpu and multi-node training, with smart local caching to minimize network bandwidth.
    * Also provides best-effort shuffling to preserve randomness when ``shuffle=True``.

    When ``batch_size`` is provided, worker indices will be constructed so that there is at most one incomplete batch at
    the end of each epoch. For example, if the DataLoader is reading over::

        samples: [0, 1, 2, 3, 4, 5, 6, 7]
        num_workers: 3
        batch_size: 2
        drop_last: True

    but ``batch_size`` is not hinted to the StreamingDataset ahead of time, then the samples will by default be assigned
    like::

        worker 0: [0, 1, 2]
        worker 1: [3, 4, 5]
        worker 2: [6, 7]

    and will be read as batches like (with samples [2] and [5] dropped as incomplete)::

        batch 0: [0, 1]
        batch 1: [3, 4]
        batch 2: [6, 7]

    The above is suboptimal because we could have dropped no samples. So when ``batch_size`` is provided as a hint, we
    assign samples like this::

        worker 0: [0, 1, 2, 3]
        worker 1: [4, 5]
        worker 2: [6, 7]

    which will be read as batches like::

        batch 0: [0, 1]
        batch 1: [4, 5]
        batch 2: [6, 7]
        batch 3: [2, 3]

    Args:
        remote (Optional[str]): Download shards from this remote path or directory.
        local (str): Download shards to this local directory for for caching.
        shuffle (bool): Whether to shuffle the samples.  Note that if ``shuffle=False``, the sample order is
            deterministic but dependent on the DataLoader's ``num_workers``.
        decoders (Dict[str, Callable[bytes, Any]]]): For each sample field you wish to read, you must provide a decoder
            to convert the raw bytes to an object.
        max_retries (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 60 sec.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader. Default:
            ``None``.
        shuffle_size (Optional[str]): Either a proportion of the dataset that is required to be loaded at once for
            optimal streaming speed, or an absolute number of shards. The string takes the format: ``0.01prop``, meaning
            1% of the dataset at a time is needed, or ``1000shards``, meaning 1000 shards at a time are needed. Lower values
            may degrade shuffle randomness, while higher values may increase cold-start lag.

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
                 remote: Optional[str],
                 local: str,
                 shuffle: bool,
                 decoders: Dict[str, Callable[[bytes], Any]],
                 max_retries: int = 2,
                 timeout: float = 60,
                 batch_size: Optional[int] = None,
                 shuffle_size: str = '2shards') -> None:

        self.remote = remote
        self.local = local
        self.shuffle = shuffle
        self.decoders = decoders
        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = batch_size

        self.compression_scheme = None
        if remote is not None:
            try:
                compression_local = self._download_file(get_compression_scheme_basename(),
                                                        wait=(dist.get_local_rank() != 0),
                                                        local_basename=get_compression_scheme_basename() + '.old')
                with open(compression_local, 'r') as fp:
                    compression_scheme = fp.read().rstrip()
                    self.compression_scheme = compression_scheme if compression_scheme != '' else None
                    if remote == local and self.compression_scheme is not None:
                        raise DatasetCompressionException('cannot decompress when remote == local')

            except FileNotFoundError:
                compression_local = os.path.join(self.local, get_compression_scheme_basename() + '.old')
                with open(compression_local, 'x') as fp:
                    fp.write('')
                pass

        # Load the index file containing the shard metadata
        # This file contains the shard and offset in bytes of each sample (for direct access).
        # Only local device 0 on each node downloads the index. All other devices wait.
        index_basename = get_index_basename(self.compression_scheme)
        index_local = self._download_file(index_basename, wait=(dist.get_local_rank() != 0))
        with open(index_local, 'rb') as fp:
            self.index = StreamingDatasetIndex.load(fp)

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock: Lock
        self._next_epoch = 0
        self._download_status = _DownloadStatus.NOT_STARTED
        self._shards_downloaded = 0
        self._download_exception: Exception

        self._shuffle_index = 0
        self._sample_count = 0
        self._restored_sample_count = 0
        self._shuffle_buffer_size = self._parse_shuffle_buffer_size(shuffle_size)
        world = get_world()
        num_nodes = world.global_num_nodes
        global_rank = world.global_node
        if shuffle:
            cipher_key = 42  # initialize using an arbitrary cipher key
            self.shuffler = BlockCipherShuffler(cipher_key, self.index, num_nodes, global_rank,
                                                int(self._shuffle_buffer_size))
            self._shard_shuffle_indices = self.shuffler.shuffle_indices
            self.index.relocate_samples(self._shard_shuffle_indices)
        else:
            N = self.index.num_shards
            self._shard_shuffle_indices = np.arange(N)[(np.arange(N) % num_nodes) == global_rank]
            self.index.relocate_samples(self._shard_shuffle_indices)
        self.num_samples = np.sum(self.index.samples_per_shard[self._shard_shuffle_indices])

    def _parse_shuffle_buffer_size(self, shuffle_buffer_size_arg: str) -> np.int64:
        if shuffle_buffer_size_arg.endswith('prop'):
            return np.int64(self.index.num_shards * float(shuffle_buffer_size_arg.strip('prop'))) + 1
        elif shuffle_buffer_size_arg.endswith('shards'):
            return np.int64(int(shuffle_buffer_size_arg.strip('shards')))
        else:
            return np.int64(1)

    def state_dict(self):
        return {'sample_count': self._sample_count, 'cipher_key': self.shuffler._cipher_key}

    def load_state_dict(self, state):
        self._restored_sample_count = state['sample_count']
        cipher_key = self.shuffler._cipher_key
        if 'cipher_key' in state:
            cipher_key = state['cipher_key']
        world = get_world()
        num_nodes = world.global_num_nodes
        global_rank = world.global_node
        self.shuffler = BlockCipherShuffler(cipher_key, self.index, num_nodes, global_rank,
                                            int(self._shuffle_buffer_size))
        self._shard_shuffle_indices = self.shuffler.shuffle_indices
        self.index.relocate_samples(self._shard_shuffle_indices)
        self.num_samples = np.sum(self.index.samples_per_shard[self._shard_shuffle_indices])

    def _download_file(self, basename: str, wait: bool = False, local_basename: Optional[str] = None) -> str:
        """Safely download a file from remote to local cache.

        Args:
            basename (str): Basename of file to download.
            wait (bool): Whether to wait for another worker to download the file.

        Returns:
            str: Local cache filename.
        """
        local_basename = local_basename if local_basename is not None else basename
        if self.remote is None:
            remote = self.remote
        else:
            remote = os.path.join(self.remote, basename)
        local = os.path.join(self.local, local_basename)
        download_or_wait(remote=remote, local=local, wait=wait, max_retries=self.max_retries, timeout=self.timeout)
        local, _ = split_compression_suffix(local)
        return local

    def download(self) -> None:
        """Downloads everything in the correct order"""

        if not hasattr(self, '_lock'):
            self._lock = Lock()

        world = get_world()

        with self._lock:
            if self._download_status != _DownloadStatus.NOT_STARTED:
                return
            self._download_status = _DownloadStatus.IN_PROGRESS

        current_shard_id = self.index.sample_shards[self._restored_sample_count]
        current_shard_index = current_shard_id

        if self.shuffle:
            if self.shuffler._cipher_key is None:
                raise ValueError('shuffling is on but no seed was specified')
            current_shard_index = self.shuffler.get_shard_index(current_shard_id)
            current_shard_index -= current_shard_index % self._shuffle_buffer_size

        shard_ids = self._shard_shuffle_indices[0:]

        for ix, shard_id in enumerate(shard_ids):
            if (ix % world.node_num_workers) != world.node_worker:
                continue
            basename = get_shard_basename(shard_id, compression_name=self.compression_scheme)
            try:
                self._download_file(basename, wait=False)
            except Exception as e:
                with self._lock:
                    self._download_status = _DownloadStatus.FAILED
                    self._download_exception = e

        with self._lock:
            if dist.get_local_rank() == 0:
                self._download_status = _DownloadStatus.DONE

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

        Do not call this directly unless the shard containing this idx has been loaded. Will crash otherwise.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: The sample.
        """
        shard = self.index.sample_id_shards[idx]
        offset = self.index.sample_shard_offsets[idx]
        size = self.index.bytes_per_sample[idx]

        basename = get_shard_basename(shard)
        shard_filename = os.path.join(self.local, basename)
        with open(shard_filename, 'rb', 0) as fp:
            fp.seek(offset)
            data = fp.read(size)

        return self._unpack_sample(data)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over all the samples in our partition.

        If not all samples have been downloaded yet, iterates over what it has while inserting the remainder into the
        sequence behind the scenes as it progresses.

        Returns:
            Iterator[Any]: Each sample.
        """
        Thread(target=self.download, daemon=True).start()
        world = get_world()
        node_num_workers = world.node_num_workers
        rank = world.node_worker
        sbs = int(self._shuffle_buffer_size)
        batch_size = 1 if self.batch_size is None else self.batch_size
        while self._sample_count < self.index.total_samples:
            # if self._sample_count < self._restored_sample_count:
            #     self._sample_count += 1
            #     yield None
            try:
                idx = self.shuffler.shuffle_sample(self._sample_count, node_num_workers, rank, sbs, batch_size, self.num_samples) \
                    if self.shuffle else self._sample_count
                yield self[idx]
                self._sample_count += 1
            except FileNotFoundError as e:
                with self._lock:
                    if self._download_status == _DownloadStatus.FAILED:
                        raise self._download_exception
                    elif self._download_status == _DownloadStatus.DONE:
                        pass
                sleep(0.25)
