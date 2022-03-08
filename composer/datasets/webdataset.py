import json
import logging
import math
import os
import subprocess
import textwrap
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple

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


def require_webdataset():
    """Hard require webdataset."""
    if not webdataset_installed:
        raise ImportError(
            textwrap.dedent("""
                Composer was installed without WebDataset support. To use WebDataset with Composer, run `pip install
                mosaicml[webdataset]`."""))


def create_webdataset_meta(split_dir: str, n_samples: int, n_shards: int) -> None:
    """Write a WebDataset meta file."""
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
                      use_tqdm: int = 1) -> None:
    """Write an entire WebDataset to a local directory, given an iterable of samples."""
    require_webdataset()
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
    create_webdataset_meta(split_dir, n_samples, n_shards)


def init_webdataset_meta_from_s3(remote: str, split: str) -> bytes:
    """Read a WebDataset meta file from S3."""
    url = f'{remote}/{split}/meta.json'
    cmd = 'aws', 's3', 'cp', url, '-'
    ret = subprocess.run(cmd, capture_output=True)
    assert not ret.stderr, 'Download failed, check your credentials?'
    return ret.stdout


def init_webdataset_meta_from_local(remote: str, split: str) -> bytes:
    """Read a WebDataset meta file from local filesystem."""
    path = f'{remote}/{split}/meta.json'
    return open(path, 'rb').read()


def init_webdataset_meta(remote: str, split: str) -> bytes:
    """Read a WebDataset meta file."""
    if remote.startswith('s3://'):
        return init_webdataset_meta_from_s3(remote, split)
    else:
        return init_webdataset_meta_from_local(remote, split)


def init_webdataset(remote: str,
                    name: str,
                    split: str,
                    cache_dir: Optional[str] = None,
                    cache_verbose: bool = False) -> Tuple[WebDataset, dict]:
    """Initialize a WebDataset with an optional local cache dir."""
    require_webdataset()
    if cache_dir:
        split_dir = os.path.join(cache_dir, name, split)
        meta_file = os.path.join(split_dir, 'meta.json')
        if os.path.exists(meta_file):
            text = open(meta_file).read()
        else:
            text = init_webdataset_meta(remote, split)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            with open(meta_file, 'wb') as out:
                out.write(text)
    else:
        split_dir = None
        text = init_webdataset_meta(remote, split)
    meta = json.loads(text)
    max_shard = meta['n_shards'] - 1
    shards = f'{{{0:05d}..{max_shard:05d}}}.tar'
    if remote.startswith('s3://'):
        urls = f'pipe: aws s3 cp {remote}/{split}/{shards} -'
    else:
        urls = f'{remote}/{split}/{shards}'
    dataset = WebDataset(urls, cache_dir=split_dir, cache_verbose=cache_verbose)
    return dataset, meta


def size_webdataset(dataset: WebDataset, n_shards: int, samples_per_shard: int, n_devices: int, workers_per_device: int,
                    batch_size: int, drop_last: bool) -> WebDataset:
    """Calculate WebDataset with_epoch() and with_length()."""
    workers_per_device = max(1, workers_per_device)

    # Ensure that shards can be split among CPU workers
    n_workers_global = n_devices * workers_per_device
    if n_shards % n_workers_global != 0:
        raise ValueError(f"n_shards={n_shards} must be divisible by n_workers_global={n_workers_global}!")

    # Set IterableDataset epoch boundary and length for DDP, PyTorch Dataloader compatability
    shards_per_worker = n_shards // n_devices // workers_per_device
    expected_samples_per_worker = samples_per_shard * shards_per_worker
    if drop_last:
        samples_per_worker = (expected_samples_per_worker // batch_size) * batch_size
        samples_per_device = samples_per_worker * workers_per_device
        samples_total = samples_per_device * n_devices
        expected_samples_total = n_shards * samples_per_shard
        if samples_total != expected_samples_total:
            log.warning(
                f"Note that 'drop_last=True' with per-CPU-worker sharding will cause an incomplete batch to be dropped at the end of ** each CPU worker's sample list **. "
                f"Given your training configuration, we have calculated this will reduce samples_per_epoch from {expected_samples_total} -> {samples_total}."
            )
    else:
        samples_per_worker = expected_samples_per_worker
        samples_per_device = samples_per_worker * workers_per_device
        samples_total = samples_per_device * n_devices
        expected_batches_per_epoch = math.ceil(samples_per_worker * workers_per_device / batch_size)
        batches_per_epoch = math.ceil(samples_per_worker / batch_size) * workers_per_device
        if batches_per_epoch != expected_batches_per_epoch:
            log.warning(
                f"Note that 'drop_last=False' with per-CPU-worker sharding will lead to multiple incomplete batches to be read from each device, ** one for each CPU worker **. "
                f"Unfortunately, the PyTorch Dataloader does not handle this situation well in its __len__ implementation, so len(dataloader) will be an underestimate of batches_per_epoch. "
                f"(See https://github.com/pytorch/pytorch/blob/3d9ec11feacd69d0ff1bffe0b25a825cdf203b87/torch/utils/data/dataloader.py#L403-L411). "
                f"Given your training configuration, we have calculated this will increase batches_per_epoch from {expected_batches_per_epoch} -> {batches_per_epoch}."
            )
    # Set epoch boundary (per CPU worker).
    # Technically not needed if shards are constructed correctly, but used for safety
    dataset = dataset.with_epoch(samples_per_worker)
    # Set IterableDataset length (per device), to be read by PyTorch Dataloader
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
        drop_last (bool): Whether to drop last.
    """
    dataset, meta = init_webdataset(remote, name, split, cache_dir, cache_verbose)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer)
    if preprocess:
        dataset = preprocess(dataset)
    return size_webdataset(dataset, meta['n_shards'], meta['samples_per_shard'], n_devices, workers_per_device,
                           batch_size, drop_last)
