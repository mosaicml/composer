# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`StreamingDataset` class, used for building streaming iterable datasets.
"""

import math
import os
from io import BytesIO
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision import transforms

from composer.datasets.streaming.download import safe_download
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict, get_index_basename,
                                                get_shard_basename)
from composer.datasets.streaming.world import get_world
from composer.utils import dist

__all__ = ['StreamingDataset', 'StreamingImageClassDataset']


class StreamingDataset(IterableDataset):
    """A sharded, streaming, iterable dataset.

    :class:`StreamingDataset` reads samples from binary `.mds` files that were written out by :class:`StreamingDatasetWriter`.

    It currently supports downloading data from etiher S3 paths or local filepaths.

    It supports multi-gpu + multi-node training, and has smart local cacheing to minimize network bandwidth.

    It also provides best-effort shuffling to preserve randomness when ``shuffle=True``.

    Args:
        remote (str): Download shards from this remote S3 path or directory.
        local (str): Download shards to this local directory for for caching.
        shuffle (bool): Whether to shuffle the samples.  Note that if `shuffle=False`, the sample order is deterministic but dependent on the DataLoader's `num_workers`.
        decoders (Dict[str, Callable[bytes, Any]]]): For each sample field you wish to read, you must provide a decoder to convert the raw bytes to an object.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 20 sec.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader.
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
                 timeout: float = 20,
                 batch_size: Optional[int] = None) -> None:

        self.remote = remote
        self.local = local
        self.shuffle = shuffle
        self.decoders = decoders
        self.timeout = timeout
        self.batch_size = batch_size

        # Load the index file containing the shard metadata, either over the
        # network or cached locally.
        # Precomputes the shard and offset in bytes of each sample (for direct
        # access).
        index_filename = self._download_if_missing(get_index_basename())
        with open(index_filename, 'rb') as fp:
            self.index = StreamingDatasetIndex.load(fp)

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock = None
        self._next_epoch = 0
        self._epoch_to_todo_ids = {}
        self._downloaded_ids = []
        self._are_all_shards_downloaded = False

    def _download_if_missing(self, basename: str) -> str:
        """Safely download a shard from remote to local cache.

        Args:
            basename (str): Basename of shard to download.

        Returns:
            str: Local cache filename.
        """
        remote = os.path.join(self.remote, basename)
        local = os.path.join(self.local, basename)
        safe_download(remote, local, timeout=self.timeout)
        return local

    def _load_shards(self, shards: List[int], part_min_id: int, part_max_id: int) -> None:
        """Load the given list of locally cached shards into the dataset.

        Every time you call __iter__ on this dataset, it registers the list of
        samples you have left, which will not be the full epoch if the dataset
        isn't finished loaded when you start training.

        Calls to _load_shards during training modify the samples remaining on
        these iterations on the fly to insert these new samples and then resort,
        making the shuffle as perfect as was possible.

        This operation takes the lock, so batch your _load_shards calls where
        possible.

        Args:
            shards (List[int]): List of shards to load.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.
        """
        # Get all samples from the given shards that fall within our partition.
        new_ids = []
        for shard in shards:
            shard_min_id = self.index.shard_begins[shard]
            shard_max_id = self.index.shard_ends[shard] - 1
            min_id = max(part_min_id, shard_min_id)
            max_id = min(part_max_id, shard_max_id)
            new_ids += list(range(min_id, max_id + 1))

        if not self._lock:
            raise RuntimeError("Attempted to use lock but lock was not created.")

        with self._lock:
            # Extend and optionally reshuffle the remaining samples of any
            # epochs we have in progress.
            if self.shuffle:
                if not self._are_all_shards_downloaded:
                    self._downloaded_ids.extend(new_ids)
                    np.random.shuffle(self._downloaded_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)
                    np.random.shuffle(todo_ids)
            else:
                if not self._are_all_shards_downloaded:
                    self._downloaded_ids.extend(new_ids)
                for todo_ids in self._epoch_to_todo_ids.values():
                    todo_ids.extend(new_ids)

    def _done_loading(self) -> None:
        """Callback on completion of loading my shards."""
        if not self._lock:
            raise RuntimeError("Attempted to use lock but lock was not created.")

        with self._lock:
            self._are_all_shards_downloaded = True

    def _download_thread(self, shards: List[int], part_min_id: int, part_max_id: int) -> None:
        """Background thread to download and assimilate missing shards.

        Args:
            shards (List[int]): The shards remaining to be downloaded.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.
        """
        shards = list(shards)
        if self.shuffle:
            np.random.shuffle(shards)
        for shard in shards:
            basename = get_shard_basename(shard)
            self._download_if_missing(basename)
            shards = [shard]
            self._load_shards(shards, part_min_id, part_max_id)
        self._done_loading()

    def _load(self) -> None:
        """Load shards."""
        # We find out num workers, and therefore num partitions, when __iter__ is called.
        # From the partition, derive our shard overlap range and exact sample range.
        world = get_world()
        part_shards, part_min_id, part_max_id = self.index.get_partition(world, self.batch_size)

        # Start downloading our part's shards in a background thread, if any are missing.
        if not self._are_all_shards_downloaded:
            thread = Thread(target=self._download_thread, args=(part_shards, part_min_id, part_max_id), daemon=True)
            thread.start()

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
        if not self._lock:
            raise RuntimeError("Attempted to use lock but lock was not created.")

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
        if not self._lock:
            raise RuntimeError("Attempted to use lock but lock was not created.")

        while True:
            with self._lock:
                todo_ids = self._epoch_to_todo_ids[epoch]
                if todo_ids:
                    return todo_ids.pop(0)
                elif self._are_all_shards_downloaded:
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
        if not self._lock:
            raise RuntimeError("Attempted to use lock but lock was not created.")

        with self._lock:
            have_full_epoch = self._are_all_shards_downloaded

        if have_full_epoch:
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
        # Lock is created here because DataLoader calls __iter__ in each worker process
        # and the lock is worker-specific
        if self._lock is None:
            self._lock = Lock()

        # Load samples
        self._load()

        for idx in self._iter_ids():
            yield self[idx]


class StreamingImageClassDataset(StreamingDataset):
    """A streaming image classification dataset, for (img, class) pairs.

       This is a subclass of :class:`StreamingDataset`.

    Args:
        remote (str): Download shards from this remote directory.
        local (str): Download shards to this local filesystem directory for reuse.
        shuffle (bool): Whether to shuffle the samples. Note that if `shuffle=False`, the sample order is deterministic but dependent on the DataLoader's `num_workers`.
        transform (Optional[Callable]): Optional input data transform for data augmentation, etc.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 20 sec.
        batch_size (Optional[int]): Hint the batch_size that will be used on each device's DataLoader.
                                    Worker indices will be constructed so that there is at most 1 incomplete batch at the end of each epoch.
    """

    def decode_image(self, data: bytes) -> Image.Image:
        """Decode the sample image.

        Args:
            data (bytes): The raw bytes.

        Returns:
            Image: PIL image encoded by the bytes.
        """
        return Image.open(BytesIO(data))

    def decode_class(self, data: bytes) -> np.int64:
        """Decode the sample class.

        Args:
            data (bytes): The raw bytes.

        Returns:
            np.int64: The class encoded by the bytes.
        """
        return np.frombuffer(data, np.int64)[0]

    def __init__(self,
                 remote: str,
                 local: str,
                 shuffle: bool,
                 transform: Optional[Callable] = None,
                 timeout: float = 20,
                 batch_size: Optional[int] = None) -> None:
        decoders = {
            'x': self.decode_image,
            'y': self.decode_class,
        }
        super().__init__(remote=remote,
                         local=local,
                         shuffle=shuffle,
                         decoders=decoders,
                         timeout=timeout,
                         batch_size=batch_size)
        self.transform = transform or transforms.ToTensor()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get the decoded and transformed (image, class) pair by ID.

        Args:
            idx (int): Sample ID.

        Returns:
            Tuple[Any, Any]: Pair of (x, y) for this sample.
        """
        obj = super().__getitem__(idx)
        x = obj['x']
        x = self.transform(x)
        y = obj['y']
        return x, y
