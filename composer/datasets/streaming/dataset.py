import os
from io import BufferedReader, BytesIO
from PIL import Image
from threading import Lock, Thread
from time import sleep
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from composer.datasets.streaming.download import safe_download
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict, get_index_basename,
                                                get_shard_basename)
from composer.utils import dist


def get_partition() -> Tuple[int, int]:
    """Get how to partition the dataset.

    Returns:
        part (int): Our partition.
        num_parts (int): Out of how many partitions.
    """
    info = get_worker_info()
    if info:
        worker_id = info.id
        workers_per_device = info.num_workers
    else:
        worker_id = 0
        workers_per_device = 1
    rank = dist.get_local_rank()
    world_size = dist.get_local_world_size()
    part = rank * workers_per_device + worker_id
    num_parts = workers_per_device * world_size
    return part, num_parts


class StreamingDataset(IterableDataset):
    """Streaming dataset."""

    def __init__(self, remote: str, local: str, split: str, shuffle: bool) -> None:
                 shuffle: bool) -> None:
        """Initialize with the given remote path and local cache.

        Loads all the samples that are available in local cache, then starts a
        background thread to download the rest during training. As samples are
        added, shuffled sample selection becomes more random.

        Args:
            remote (str): Download shards from this remote directory.
            local (str): Download shards to this local filesystem directory for reuse.
            split (str): Which dataset split.
            shuffle (bool): Whether to shuffle the samples.
        """
        self.remote = remote
        self.local = local
        self.split = split
        self.shuffle = shuffle

        # Load the index file containing the shard metadata, either over the
        # network or cached locally.
        # Precomputes the shard and offset in bytes of each sample (for direct
        # access).
        local = self._download_if_missing(get_index_basename())
        self.index = StreamingDatasetIndex.load(open(local, 'rb'))

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock = Lock()
        self._files: List[Optional[BufferedReader]] = [None] * self.index.num_shards
        self._next_epoch_key = 0
        self._key_to_epoch = {}
        self._loaded_epoch = []
        self._is_loading_complete = False

    def __del__(self) -> None:
        """The destructor closes any file handles left open to read shards."""
        for fp in self._files:
            if fp:
                fp.close()

    def _download_if_missing(self, basename: str) -> str:
        """Safely download a shard from remote to local cache.

        Args:
            basename (str): Basename of shard to download.

        Returns:
            str: Local cache filename.
        """
        remote = os.path.join(self.remote, self.split, basename)
        local = os.path.join(self.local, self.split, basename)
        safe_download(remote, local)
        return local

    def _load_shards(self, shards: Sequence[int], part_min_id: int, part_max_id: int) -> None:
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
            shards (Sequence[int]): List of shards to load.
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

        with self._lock:
            # Keep open a file handle to each loaded shard for fast access.
            for shard in shards:
                basename = get_shard_basename(shard)
                filename = os.path.join(self.local, self.split, basename)
                assert self._files[shard] is None, 'Open file handle already exists!'
                self._files[shard] = open(filename, 'rb')

            # Extend and reshuffle the remaining samples of any epochs we have
            # going on.
            if self.shuffle:
                if not self._is_loading_complete:
                    self._loaded_epoch.extend(new_ids)
                    np.random.shuffle(self._loaded_epoch)

                for epoch in self._key_to_epoch.values():
                    epoch.extend(new_ids)
                    np.random.shuffle(epoch)
            else:
                if not self._is_loading_complete:
                    self._loaded_epoch.extend(new_ids)
                for epoch in self._key_to_epoch.values():
                    epoch.extend(new_ids)

    def _load_shards_if_downloaded(self, shards: Sequence[int], part_min_id: int, part_max_id: int) -> List[int]:
        """Load any of the given shards that are already present in the cache, returning the missing shards.

        Args:
            shards (Sequence[int]): The shards to attempt to load.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.

        Returns:
            list of int: The shards that remain to be loaded.
        """
        downloaded = []
        missing = []
        for shard in sorted(shards):
            basename = get_shard_basename(shard)
            local = os.path.join(self.local, self.split, basename)
            if os.path.exists(local):
                downloaded.append(shard)
            else:
                missing.append(shard)
        if downloaded:
            self._load_shards(downloaded, part_min_id, part_max_id)
        return missing

    def _done_loading(self) -> None:
        """Callback on completion of loading my shards."""
        with self._lock:
            self._is_loading_complete = True

    def _download_thread(self, shards: Sequence[int], part_min_id: int, part_max_id: int) -> None:
        """Background thread to download and assimilate missing shards.

        Args:
            shards (list of int): The shards remaining to be downloaded.
            part_min_id (int): Minimum sample ID of this partition.
            part_max_id (int): Maximum sample ID of this partition.
        """
        shards = list(shards)
        if self.shuffle:
            np.random.shuffle(shards)
        for shard in shards:
            basename = get_shard_basename(shard)
            self._download_if_missing(basename)
            shards = shard,
            self._load_shards(shards, part_min_id, part_max_id)
        self._done_loading()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Dataset length.
        """
        return self.index.total_samples

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index, assuming its shard is loaded.

        Do not call this directly unless all shards have been loaded. Will crash
        if the shard is not loaded.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: The sample.
        """
        shard = self.index.sample_shards[idx]
        offset = self.index.sample_shard_offsets[idx]
        size = self.index.bytes_per_sample[idx]
        fp = self._files[shard]
        assert fp is not None, 'Tried to __getitem__ a sample that was not loaded.'
        fp.seek(offset)
        data = fp.read(size)
        return bytes_to_sample_dict(data, self.index.fields)

    def _new_growing_epoch(self) -> int:
        """Start a new growing epoch, in which we own the sample sequence because it grows.

        Returns:
            int: The epoch key, a handle which is given back to the caller.
        """
        with self._lock:
            key = self._next_epoch_key
            self._next_epoch_key += 1
            epoch = list(self._loaded_epoch)
            self._key_to_epoch[key] = epoch
        return key

    def _next_id(self, key: int) -> Optional[int]:
        """Get next sample of the growing epoch given by key, or None if done.

        If we are currently out of samples but not finished downloading the
        shards, blocks until it has new samples.

        Args:
            key (int): The epoch key, a handle for this sequence.

        Returns:
            int: ID of next sample.
        """
        while True:
            with self._lock:
                epoch = self._key_to_epoch[key]
                if epoch:
                    return epoch.pop()
                elif self._is_loading_complete:
                    del self._key_to_epoch[key]
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
            have_full_epoch = self._is_loading_complete

        if have_full_epoch:
            epoch = list(self._loaded_epoch)
            if self.shuffle:
                np.random.shuffle(epoch)
            for idx in epoch:
                yield idx
        else:
            key = self._new_growing_epoch()
            while True:
                idx = self._next_id(key)
                if idx is None:
                    break
                yield idx

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        """Iterate over all the samples in our partition.

        If not all samples have been downloaded yet, iterates over what it has
        while inserting the remainder into the sequence behind the scenes as it
        progresses.

        Returns:
            Iterator[Tuple[Tensor, Tensor]]: Each sample.
        """
        # We find out num workers, and therefore num partitions, when __iter__ is called.
        # From the partition, derive our shard overlap range and exact sample range.
        part, num_parts = get_partition()
        todo_shards, part_min_id, part_max_id = self.index.get_partition_shards_and_samples(part, num_parts)

        # Preload all of our shards that are already available in the cache.
        todo_shards = self._load_shards_if_downloaded(todo_shards, part_min_id, part_max_id)

        # Start downloading our missing shards in a background thread, if there are any.
        if todo_shards:
            thread = Thread(target=self._download_thread, args=(todo_shards, part_min_id, part_max_id), daemon=True)
            thread.start()
        else:
            self._done_loading()

        # Iterate over the samples we have while the rest are begin loaded.
        for idx in self._iter_ids():
            yield self[idx]


class StreamingVisionDataset(StreamingDataset, VisionDataset):
    """Streaming vision dataset.

    Analogous to `torchvision.datasets.VisionDataset` plus streaming, but this
    is an IterableDataset, so do not naively rely on __getitem__.
    """

    def __init__(self, root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """Initialize with the same API as VisionDataset.

        Args:
            root (str): Root directory of dataset.
            transforms (Optional[Callable]): A function/transforms that takes in
                an image and a label and returns the transformed versions of
                both.
            transform (Optional[Callable]): A function/transform that  takes in
                an PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (Optional[Callable]): A function/transform that
                takes in the target and transforms it.
        """
        self.root = root
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx: int) -> Any:
        """Get the sample at the index, assuming its shard is loaded.

        Do not call this directly unless all shards have been loaded. Will crash
        if the shard is not loaded.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: The sample.
        """
        obj = super()[idx]

        # The "data" field is a PIL image.
        x = obj['data']
        x = Image.open(BytesIO(x))

        # The "target" field is an int.
        y = obj['target']
        y = np.frombuffer(y, np.int64)[0]
        y = int(y)

        # Do the transform (joint or separate).
        if self.transforms:
            x, y = self.transforms(x, y)
        else:
            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)

        return x, y
