import json
import logging
import math
import os
import subprocess
import textwrap
from random import shuffle
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from tqdm import tqdm

if TYPE_CHECKING:
    from webdataset import WebDataset

try:
    from webdataset import ShardWriter, WebDataset
    from wurlitzer import pipes
    webdataset_installed = True
except ImportError:
    webdataset_installed = False

log = logging.getLogger(__name__)


def _require_webdataset():
    """Hard require webdataset."""
    if not webdataset_installed:
        raise ImportError(
            textwrap.dedent("""
                Composer was installed without WebDataset support. To use WebDataset with Composer, run `pip install
                mosaicml[webdataset]`."""))


def _create_webdataset_meta(split_dir: str, n_samples: int, n_shards: int) -> None:
    """Write a WebDataset meta file.

    Args:
        split_dir (str): Directory to save the JSON file into.
        n_samples (int): Number of samples in this split.
        n_shards (int): Number of shards in this split.
    """
    samples_per_shard = n_samples // n_shards
    n_leftover = n_samples % samples_per_shard
    obj = {
        'n_shards': n_shards,
        'samples_per_shard': samples_per_shard,
        'n_leftover': n_leftover,
    }
    filename = os.path.join(split_dir, 'meta.json')
    json.dump(obj, open(filename, 'w'), sort_keys=True)


def create_webdataset(samples: Iterable[Dict[str, Any]],
                      dataset_dir: str,
                      split: str,
                      n_samples: int,
                      n_shards: int,
                      use_tqdm: Union[bool, int] = True) -> None:
    """Write an entire WebDataset to a local directory, given an iterable of samples.

    Args:
        samples (iterable of dict): Each dataset sample.
        dataset_dir (str): Output dataset directory.
        split (str): Dataset split.
        n_samples (int): Number of samples in dataset.
        n_shards (int): Number of full shards to write (may write a leftovers shard).
        use_tqdm (bool): Whether to show progress with tqdm.
    """
    _require_webdataset()
    split_dir = os.path.join(dataset_dir, split)
    os.makedirs(split_dir)
    pattern = os.path.join(split_dir, '%05d.tar')
    samples_per_shard = n_samples // n_shards
    with pipes():
        out = ShardWriter(pattern, maxcount=samples_per_shard)
        out.verbose = 0
    if use_tqdm:
        samples = tqdm(samples, total=n_samples, leave=False)
    for sample in samples:
        out.write(sample)
    out.close()
    _create_webdataset_meta(split_dir, n_samples, n_shards)


def _find_samples(split_dirname):
    """Collect and shuffle sample as pairs of (image filename, class).

    Args:
        split_dirname (str): Dataset split directory.

    Returns:
        Shuffled list of (image filename, class).
    """
    pairs = []
    for cls, basename in enumerate(sorted(os.listdir(split_dirname))):
        class_dirname = os.path.join(split_dirname, basename)
        for basename in sorted(os.listdir(class_dirname)):
            sample_filename = os.path.join(class_dirname, basename)
            pairs.append((sample_filename, cls))
    shuffle(pairs)
    return pairs


def _each_sample(pairs: List[Tuple[str, int]]) -> Iterable[Dict[str, Any]]:
    """Generator over each dataset sample.

    Args:
        pairs (list): List of pairs of (image filename, class ID).

    Yields:
        Sample dicts.
    """
    for idx, (img_file, cls) in enumerate(pairs):
        img = open(img_file, 'rb').read()
        yield {
            '__key__': f'{idx:05d}',
            'jpg': img,
            'cls': cls,
        }


def create_webdatasets_from_image_folder(in_root: str,
                                         out_root: str,
                                         n_shards: int,
                                         use_tqdm: Union[bool, int] = True) -> None:
    """Given a directory tree of classified images, create a WebDataset per dataset split.

    Directory tree format: (path to dataset)/(split name)/(class name)/(image file).

    Args:
        in_root (str): Input dataset root.
        out_root (str): Output WebDataset root.
        n_shards (int): Number of full shards to write (may write a leftovers shard).
        use_tqdm (bool): Whether to show progress with tqdm.
    """
    for split in sorted(os.listdir(in_root)):
        in_dir = os.path.join(in_root, split)
        pairs = _find_samples(in_dir)
        create_webdataset(_each_sample(pairs), out_root, split, len(pairs), n_shards, use_tqdm)


def _init_webdataset_meta_from_s3(remote: str, split: str) -> bytes:
    """Read a WebDataset meta file from S3.

    Args:
        remote (str): S3 bucket or S3 bucket directory.
        split (str): Dataset split.
    """
    url = f'{remote}/{split}/meta.json'
    cmd = 'aws', 's3', 'cp', url, '-'
    ret = subprocess.run(cmd, capture_output=True)
    assert not ret.stderr, 'Download failed, check your credentials?'
    return ret.stdout


def _init_webdataset_meta_from_local(remote: str, split: str) -> bytes:
    """Read a WebDataset meta file from local filesystem.

    Args:
        remote (str): Local filesystem directory.
        split (str): Dataset split.
    """
    path = f'{remote}/{split}/meta.json'
    return open(path, 'rb').read()


def _init_webdataset_meta(remote: str, split: str) -> bytes:
    """Read a WebDataset meta file.

    Args:
        remote (str): Dataset directory (S3 bucket or local dir).
        split (str): Dataset split.
    """
    if remote.startswith('s3://'):
        return _init_webdataset_meta_from_s3(remote, split)
    else:
        return _init_webdataset_meta_from_local(remote, split)


def _init_webdataset(remote: str,
                     name: str,
                     split: str,
                     cache_dir: Optional[str] = None,
                     cache_verbose: bool = False) -> Tuple[WebDataset, dict]:
    """Initialize a WebDataset with an optional local cache dir.

    Args:
        remote (str): Dataset directory (S3 bucket or local dir).
        name (str): Name of this dataset, used to locate dataset in local cache.
        split (str): Dataset split.
        cache_dir (str, optional): Root directory of local filesystem cache.
        cache_verbose (bool): WebDataset caching verbosity.

    Returns:
        dataset (WebDataset): The webdataset object for streaming.
        meta (dict): Dataset sample/shard statistics.
    """
    _require_webdataset()
    if cache_dir:
        split_dir = os.path.join(cache_dir, name, split)
        meta_file = os.path.join(split_dir, 'meta.json')
        if os.path.exists(meta_file):
            text = open(meta_file).read()
        else:
            text = _init_webdataset_meta(remote, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            with open(meta_file, 'wb') as out:
                out.write(text)
    else:
        split_dir = None
        text = _init_webdataset_meta(remote, split)
    meta = json.loads(text)
    max_shard = meta['n_shards'] - 1
    shards = f'{{{0:05d}..{max_shard:05d}}}.tar'
    if remote.startswith('s3://'):
        urls = f'pipe: aws s3 cp {remote}/{split}/{shards} -'
    else:
        urls = f'{remote}/{split}/{shards}'
    dataset = WebDataset(urls, cache_dir=split_dir, cache_verbose=cache_verbose)
    return dataset, meta


def _size_webdataset(dataset: WebDataset, n_shards: int, samples_per_shard: int, n_devices: int,
                     workers_per_device: int, batch_size: int, drop_last: bool) -> WebDataset:
    """Set IterableDataset epoch boundary and length for DDP, PyTorch DataLoader compatability.

    Note: 'drop_last=True' with per-CPU-worker sharding will cause an incomplete batch to be dropped at the end of each
    CPU worker's sample list. Total samples dropped across all workers may sum to more than one batch.

    Note: 'drop_last=False' with per-CPU-worker sharding will lead to multiple incomplete batches being read from each
    device, one for each CPU worker. Unfortunately, the PyTorch DataLoader does not handle this situation well in its
    __len__ implementation, so len(dataloader) will be an underestimate of batches_per_epoch.

    Calculation:
                                        shards
        shards per worker = ------------------------------
                             devices * workers per device

        samples per worker = samples per shard * shards per worker

        If drop last,
            samples per worker = (samples per worker // batch size) * batch size

        samples per device = samples per worker * workers per device

        samples per epoch = samples per device * devices

    Args:
        dataset (WebDataset):
        n_shards (int): Number of full shards.
        samples_per_shard (int): Number of samples per webdataset shard.
        n_devices (int): Number of devices.
        workers_per_device (int): Number of workers per device.
        batch_size (int): Batch size.
        drop_last (bool): Whether to drop partial last batches.
    """
    workers_per_device = max(1, workers_per_device)

    # Ensure that shards can be split among CPU workers
    n_workers_global = n_devices * workers_per_device
    if n_shards % n_workers_global != 0:
        raise ValueError(f"n_shards={n_shards} must be divisible by n_workers_global={n_workers_global}!")

    shards_per_worker = n_shards // n_devices // workers_per_device
    expected_samples_per_worker = samples_per_shard * shards_per_worker
    if drop_last:
        samples_per_worker = (expected_samples_per_worker // batch_size) * batch_size
        samples_per_device = samples_per_worker * workers_per_device
        samples_per_epoch = samples_per_device * n_devices
        expected_samples_per_epoch = n_shards * samples_per_shard
        if samples_per_epoch != expected_samples_per_epoch:
            log.warning(
                f"Note that 'drop_last=True' with per-CPU-worker sharding will cause an incomplete batch to be dropped at the end of ** each CPU worker's sample list **. "
                f"Given your training configuration, we have calculated this will reduce samples_per_epoch from {expected_samples_per_epoch} to {samples_per_epoch}."
            )
    else:
        samples_per_worker = expected_samples_per_worker
        samples_per_device = samples_per_worker * workers_per_device
        samples_per_epoch = samples_per_device * n_devices
        expected_batches_per_epoch = math.ceil(samples_per_worker * workers_per_device / batch_size)
        batches_per_epoch = math.ceil(samples_per_worker / batch_size) * workers_per_device
        if batches_per_epoch != expected_batches_per_epoch:
            log.warning(
                f"Note that 'drop_last=False' with per-CPU-worker sharding will lead to multiple incomplete batches being read from each device, ** one for each CPU worker **. "
                f"Unfortunately, the PyTorch DataLoader does not handle this situation well in its __len__ implementation, so len(dataloader) will be an underestimate of batches_per_epoch. "
                f"(See https://github.com/pytorch/pytorch/blob/3d9ec11feacd69d0ff1bffe0b25a825cdf203b87/torch/utils/data/dataloader.py#L403-L411). "
                f"Given your training configuration, we have calculated this will increase batches_per_epoch from {expected_batches_per_epoch} -> {batches_per_epoch}."
            )
    # Set epoch boundary (per CPU worker).
    # Technically not needed if shards are constructed correctly, but used for safety
    dataset = dataset.with_epoch(samples_per_worker)
    # Set IterableDataset length (per device), to be read by PyTorch DataLoader
    return dataset.with_length(samples_per_device)


def load_webdataset(remote: str, name: str, split: str, cache_dir: Optional[str], cache_verbose: bool, shuffle: bool,
                    shuffle_buffer: int, preprocess, n_devices: int, workers_per_device: int, batch_size: int,
                    drop_last: bool):
    """Load WebDataset from remote, optionally caching, with the given preprocessing and batching.

    Args:
        remote (str): Remote path (either an s3:// url or a directory on local filesystem).
        name (str): Name of this dataset, used to locate dataset in local cache.
        cache_dir (str, optional): Root directory of local filesystem cache.
        cache_verbose (bool): WebDataset caching verbosity.
        shuffle (bool): Whether to shuffle samples.
        shuffle_buffer (int): How many samples to buffer when shuffling.
        preprocess (Callable): What transformations to apply to the samples, as WebDataset iterator(s).
        n_devices (int): Number of devices.
        workers_per_device (int): Number of workers per device.
        batch_size (int): Batch size.
        drop_last (bool): Whether to drop partial last batches.
    """
    dataset, meta = _init_webdataset(remote, name, split, cache_dir, cache_verbose)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    if preprocess:
        dataset = preprocess(dataset)
    return _size_webdataset(dataset, meta['n_shards'], meta['samples_per_shard'], n_devices, workers_per_device,
                            batch_size, drop_last)
