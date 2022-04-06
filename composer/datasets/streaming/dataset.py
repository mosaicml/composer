import os
from io import BufferedReader
from threading import Lock, Thread
from time import sleep
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import IterableDataset

from composer.datasets.streaming.download import safe_download
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict, get_index_basename,
                                                get_shard_basename)


class StreamingDataset(IterableDataset):
    """StreamingDataset."""

    def __init__(self, remote: str, local: str, split: str, transform: Callable, target_transform: Callable,
                 shuffle: bool) -> None:
        """Initialize with the given remote path and local cache.

        Loads all the samples that are available in local cache, then starts a background thread to download the rest
        during training. As samples are added, shuffled sample selection becomes more random.

        Args:
            remote (str): Download shards from this remote directory.
            local (str): Download shards to this local filesystem directory for reuse.
            split (str): Which dataset split.
            transform (callable, optional): X transformation.
            target_transform (callable, optional): Y transformation.
            shuffle (bool): Whether to shuffle the samples.
        """
        self.remote = remote
        self.local = local
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle

        # First, every worker loads the index file (one downloads/caches while the others poll).
        local = self._download_if_missing(get_index_basename())
        fp = open(local, 'rb')
        self.index = StreamingDatasetIndex.load(fp)

        # Given that, precompute shard and byte offset of all our samples, giving us which shards we need to load.
        self.sample_shards, self.sample_offsets = self.index.locate_samples()
        todo_shards = sorted(set(self.sample_shards))

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock = Lock()
        self._files: List[Optional[BufferedReader]] = [None] * self.index.num_shards
        self._next_epoch_key = 0
        self._key2epoch = {}
        self._loaded_epoch = []
        self._is_loading_complete = False

        # Preload all of our shards that are already available in the cache.
        #
        # If we are processing samples sequentially (i.e., not shuffled), we need to download and load their shards
        # sequentially. This is covered in the background thread below. No help to preload shards.
        if self.shuffle:
            todo_shards = self._load_shards_if_downloaded(todo_shards)

        # Start downloading our missing shards in a background thread, if there are any.
        if todo_shards:
            thread = Thread(target=self._download_thread, args=(todo_shards,), daemon=True)
            thread.start()
        else:
            self._done_loading_shards()

    def __del__(self):
        for fp in self._files:
            if fp:
                fp.close()

    def _download_if_missing(self, basename: str) -> str:
        """Safely download a shard from remote to local cache, returning local filename.

        Args:
            basename (str): Basename of shard to download.

        Returns:
            str: Local cache filename.
        """
        remote = os.path.join(self.remote, self.split, basename)
        local = os.path.join(self.local, self.split, basename)
        safe_download(remote, local)
        return local

    def _do_load_shards(self, shards: List[int]) -> None:
        """Assimilate the given list of locally cached shards into the dataset.

        Every time you call __iter__ on this dataset, it registers the list of samples you have left, which will not be
        the full epoch if the dataset isn't finished loaded when you start training.

        Calls to _do_load_shards during training modify the samples remaining on these iterations on the fly to
        insert these new samples and then resort, making the shuffle as perfect as was possible.

        This operation takes the lock, so try to group shards to load as best as possible.

        Args:
            shards (list of int): List of shards to load.
        """
        new_ids = []
        for shard in shards:
            begin, end = self.index.get_shard_sample_range(shard)
            new_ids += list(range(begin, end))

        with self._lock:
            for shard in shards:
                basename = get_shard_basename(shard)
                filename = os.path.join(self.local, self.split, basename)
                self._files[shard] = open(filename, 'rb')

            if self.shuffle:
                self._loaded_epoch.extend(new_ids)
                np.random.shuffle(self._loaded_epoch)
                for epoch in self._key2epoch.values():
                    epoch.extend(new_ids)
                    np.random.shuffle(epoch)
            else:
                self._loaded_epoch.reverse()
                self._loaded_epoch.extend(new_ids)
                self._loaded_epoch.reverse()
                for epoch in self._key2epoch.values():
                    epoch.reverse()
                    epoch.extend(new_ids)
                    epoch.reverse()

    def _load_shards_if_downloaded(self, shards: List[int]) -> List[int]:
        """Load any of the given shards that are already present in the cache, returning the missing shards.

        Args:
            shards (list of int): The shards to attempt to load.

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
            self._do_load_shards(downloaded)
        return missing

    def _done_loading_shards(self) -> None:
        """Callback on completion of loading my shards."""
        with self._lock:
            self._is_loading_complete = True

    def _download_thread(self, shards: List[int]) -> None:
        """Background thread to download and assimilate missing shards.

        Args:
            shards (list of int): The shards remaining to be downloaded.
        """
        shards = list(shards)
        if self.shuffle:
            np.random.shuffle(shards)
        for shard in shards:
            basename = get_shard_basename(shard)
            self._download_if_missing(basename)
            self._do_load_shards([shard])
        self._done_loading_shards()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Dataset length.
        """
        return self.index.total_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get the sample at the index, assuming its shard is loaded.

        Do not call this directly unless all shards have been loaded. Will crash if the shard is not loaded.

        Args:
            idx (int): Sample ID.

        Returns:
            x (tensor): The input tensor.
            y (tensor): The output tensor.
        """
        shard = self.sample_shards[idx]
        offset = self.sample_offsets[idx]
        size = self.index.bytes_per_sample[idx]
        fp = self._files[shard]
        assert fp is not None, 'Tried to __getitem__ a sample that was not loaded.'
        fp.seek(offset)
        data = fp.read(size)
        obj = bytes_to_sample_dict(data, self.index.fields)
        x = self.transform(obj['x'])
        y = self.target_transform(obj['y'])
        return x, y

    def _new_growing_epoch(self) -> int:
        """Start a new growing epoch, in which we own the sample sequence because it grows.

        Returns:
            int: The epoch key, a handle for this sequence which is given back to the caller.
        """
        with self._lock:
            key = self._next_epoch_key
            self._next_epoch_key += 1
            epoch = list(self._loaded_epoch)
            self._key2epoch[key] = epoch
        return key

    def _next_id(self, key: int) -> Optional[int]:
        """Get the next sample of the growing epoch referenced by the given epoch key, or None if done.

        If we are currently out of samples but not finished downloading the shards, blocks until it has new samples.

        Args:
            key (int): The epoch key, a handle for this sequence.

        Returns:
            int: ID of next sample.
        """
        while True:
            with self._lock:
                epoch = self._key2epoch[key]
                if epoch:
                    return epoch.pop()
                elif self._is_loading_complete:
                    del self._key2epoch[key]
                    return None
                else:
                    pass
            sleep(0.25)

    def _iter_ids(self) -> Iterator[int]:
        """Get an iterator over all our sample IDs.

        Returns:
            iterator over pairs of tensors: Each sample ID.
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
        """Get an iterator over all our samples.

        If not all samples have been downloaded yet, iterates over what it has while inserting the remainder into the
        sequence behind the scenes as it progresses.

        Returns:
            iterator over pairs of tensors: Each sample.
        """
        for idx in self._iter_ids():
            yield self[idx]
