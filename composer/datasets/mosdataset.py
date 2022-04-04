import boto3
import numpy as np
import os
import shutil
from threading import Lock, Thread
from time import sleep
from torch import Tensor
from torch.utils.data import IterableDataset
from tqdm import tqdm
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple


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
            fields (list of str): The names of the sample's fields in order.
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


def download(remote, local) -> None:
    """Universal downloader.

    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
    """
    dirname = os.path.dirname(local)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if remote.startswith('s3://'):
        remote = remote[5:]
        idx = remote.index('/')
        bucket = remote[:idx]
        path = remote[idx + 1:]
        s3 = boto3.client('s3')
        s3.download_file(bucket, path, local)
    else:
        shutil.copy(remote, local)


class MosaicDataset(IterableDataset):
    """MosaicDataset."""

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 transform: Optional[Callable],
                 target_transform: Optional[Callable],
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
        local = self._download_if_missing('index.mds')
        fp = open(local, 'rb')
        self.index = MosaicDatasetIndex.load(fp)

        # Given that, precompute shard and byte offset of all our samples, giving us which shards we need to load.
        self.sample_shards, self.sample_offsets = self.index.locate_samples()
        todo_shards = sorted(set(self.sample_shards))

        # Fields, protected by the lock, relating to loading shards in the background.
        self._lock = Lock()
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

    def _wait_for_download(self, filename: str) -> None:
        """Block until a shard download completes.

        Args:
            filename (str): Path to file.
        """
        i = 0
        while True:
            if os.path.exists(filename):
                return
            if 4 <= i and not i % 4:
                print('Waiting for download:', filename)
            sleep(0.25)
            i += 1

    def _download_if_missing(self, basename: str) -> str:
        """Safely download a shard from remote to local cache, returning local filename.

        Args:
            basename (str): Basename of shard to download.

        Returns:
            str: Local cache filename.
        """
        # If we already have the file cached locally, we're done.
        local = os.path.join(self.local, self.split, basename)
        if os.path.exists(local):
            return local

        # Else if someone else is currently downloading the shard, wait for that download to complete.
        local_tmp = local + '.tmp'
        if os.path.exists(local_tmp):
            self._wait_for_download(local)
            return local

        # Else if no one is downloading it, mark as in progress, then do the download ourself.
        local_dir = os.path.join(self.local, self.split)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        with open(local_tmp, 'w') as out:
            out.write('')
        remote = os.path.join(self.remote, self.split, basename)
        download(remote, local_tmp)
        os.rename(local_tmp, local)
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
            basename = '%05d.mds' % shard
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
            basename = '%05d.mds' % shard
            self._download_if_missing(basename)
            self._do_load_shards([shard])
        self._done_loading_shards()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Dataset length.
        """
        return self.index.num_samples

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
        basename = '%05d.mds' % shard
        filename = os.path.join(self.local, self.split, basename)
        fp = open(filename, 'rb')
        fp.seek(offset)
        data = fp.read(size)
        fp.close()
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
