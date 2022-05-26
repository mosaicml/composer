from argparse import ArgumentParser
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
import numpy as np
from numpy.typing import NDArray
import os
import psutil
from threading import Lock, Thread
from time import sleep
import torch
from torch import distributed as dist
from torch.utils.data import get_worker_info, IterableDataset
from typing import Any, Callable, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from composer.datasets.streaming.format import (
    bytes_to_sample_dict, get_index_basename, get_shard_basename, StreamingDatasetIndex)
from composer.utils import dist as c_dist


def get_has_shard_basename():
    return 'has_shard.u8'


def parse_args():
    args = ArgumentParser()
    args.add_argument('--rank', type=int, required=True)
    args.add_argument('--world_size', type=int, required=True)
    args.add_argument('--local_world_size', type=int, default=8)
    args.add_argument('--master_addr', type=str, required=True)
    args.add_argument('--master_port', type=int, required=True)
    return args.parse_args()


class World(object):
    def __init__(self, rank, world_size, local_world_size):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % local_world_size
        self.local_world_size = local_world_size
        self.device = torch.device('cuda:%d' % self.local_rank)

        self.leader = 0
        self.is_leader = self.rank == self.leader
        self.is_local_leader = not self.local_rank
        self.is_ng_local_leader = self.is_local_leader and not self.is_leader

        self.node = self.rank // local_world_size
        self.num_nodes = self.world_size // local_world_size


def _get_root_python_pid() -> int:
    pid = os.getpid()
    ppid = os.getppid()
    proc = psutil.Process(ppid)
    while proc.name().startswith('python'):
        pid = proc.pid
        ppid = proc.ppid()
        proc = psutil.Process(ppid)
    return pid


def _send_broadcast(world: World, data: bytes) -> None:
    """Send broadcast to all other ranks.

    Args:
        world (World): World info.
        data (bytes): The bytes to broadcast.
    """
    size = torch.tensor([len(data)], dtype=torch.int64, device=world.device)
    dist.broadcast(size, world.rank)

    arr = np.frombuffer(data, np.uint8)
    tensor = torch.tensor(arr, dtype=torch.uint8, device=world.device)
    dist.broadcast(tensor, world.rank)


def _recv_broadcast(world: World, source: int) -> bytes:
    """Receive broadcast from the specified rank.

    Args:
        world (World): World info.
        source (int): Rank of broadcaster.

    Returns:
        bytes: The bytes that were broadcasted.
    """
    size = torch.empty(1, dtype=torch.int64, device=world.device)
    dist.broadcast(size, source)
    size = int(size)

    data = torch.empty(size, dtype=torch.uint8, device=world.device)
    dist.broadcast(data, source)
    return data.cpu().numpy().tobytes()


def _get_seed(seed: Optional[int] = None) -> int:
    """Get the seed.

    Args:
        seed (Optional[int]): Optional seed.

    Returns:
        int: Seed (randomly generated if not provided).
    """
    if seed is None:
        seed = int(np.random.choice(1 << 31))
    return seed


def _distribute_seed(world: World, seed: Optional[int] = None):
    """Initialize and distribute the same seed from rank zero to all processes.

    Args:
        world (World): World info.
        seed (Optional[int]): Optional seed.

    Returns:
        int: Seed from rank zero.
    """
    if world.world_size == 1:
        seed = _get_seed(seed)
    elif world.is_leader:
        seed = _get_seed(seed)
        data = torch.tensor([seed], dtype=torch.int32, device=world.device)
        dist.broadcast(data, world.leader)
    else:
        data = torch.empty(1, dtype=torch.int32, device=world.device)
        dist.broadcast(data, world.leader)
        seed = int(data)
    return seed


def _download_from_s3(remote: str, local: str, retry: int, timeout: float) -> None:
    """Download a file from S3.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    import boto3
    from botocore.config import Config
    obj = urlparse(remote)
    assert obj.scheme == 's3'
    config = Config(connect_timeout=timeout, read_timeout=timeout, retries={'total_max_attempts': retry + 1})
    s3 = boto3.client('s3', config=config)
    s3.download_file(obj.netloc, obj.path[1:], local)


def _download_from_local(remote: str, local: str) -> None:
    """Download a file from the local filesystem.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
    """
    import shutil
    shutil.copy(remote, local)


def _download(remote: str, local: str, retry: int, timeout: float):
    """Download a file from a remote filesystem.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    if os.path.exists(local):
        return
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if remote.startswith('s3://'):
        _download_from_s3(remote, local, retry, timeout)
    else:
        _download_from_local(remote, local)


def _download_index(remote: str, local: str, retry: int, timeout: float):
    """Download a StreamingDatasetIndex file from a remote filesystem.

    Args:
        remote (str): Remote file path.
        local (str): Local file path.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    basename = get_index_basename()
    remote_filename = os.path.join(remote, basename)
    local_filename = os.path.join(local, basename)
    _download(remote_filename, local_filename, retry, timeout)
    return open(local_filename, 'rb').read()


def _save_index(index_data: bytes, local: str) -> None:
    """Save serialized StreamingDatasetIndex to a local streaming dataset directory.

    Args:
        index_data (bytes): Bytes to write.
        local (str): Local directory path.
    """
    filename = os.path.join(local, get_index_basename())
    if os.path.isfile(filename):
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'xb') as out:
        out.write(index_data)


def _download_and_distribute_index(world: World, remote: str, local: str, retry: int,
                                   timeout: float) -> StreamingDatasetIndex:
    """Download and distribute a streaming dataset index.

    Args:
        world (World): World info.
        remote (str): Remote file path.
        local (str): Local file path.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.

    Returns:
        StreamingDatasetIndex: The index.
    """
    if world.world_size == 1:
        index_data = _download_index(remote, local, retry, timeout)
    elif world.is_leader:
        index_data = _download_index(remote, local, retry, timeout)
        _send_broadcast(world, index_data)
    elif world.is_ng_local_leader:
        index_data = _recv_broadcast(world, world.leader)
        _save_index(index_data, local)
    else:
        index_data = _recv_broadcast(world, world.leader)
    return StreamingDatasetIndex.loads(index_data)


def _list_shards(local: str, num_shards: int) -> NDArray[np.uint8]:
    """Get which shards have been downloaded, saving array to file for fast lookup.

    This file is then polled by all the workers for shards to arrive.

    Args:
        local (str): Local dataset directory.
        num_shards (int): Number of dataset shards to expect.

    Returns:
        NDArray[np.uint8]: Whether downloaded (byte per file).
    """
    has_shard = np.empty(num_shards, np.uint8)
    for shard in range(num_shards):
        filename = os.path.join(local, get_shard_basename(shard))
        has_shard[shard] = os.path.isfile(filename)
    filename = os.path.join(local, get_has_shard_basename())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    has_shard.tofile(filename)
    return has_shard


def _mp_list_shards(world: World, local: str, num_shards: int) -> NDArray[np.uint8]:
    """Get which shards have been downloaded (called by all ranks).

    Args:
        world (World): World info.
        local (str): Local dataset directory.
        num_shards (int): Number of dataset shards to expect.

    Returns:
        NDArray[np.uint8]: Whether downloaded (byte per file).
    """
    has_shard = np.zeros(num_shards, np.uint8)
    if world.is_local_leader:
        has_shard = _list_shards(local, num_shards)
    dist.barrier()
    if not world.is_local_leader:
        has_shard_filename = os.path.join(local, get_has_shard_basename())
        has_shard = np.fromfile(has_shard_filename, np.uint8)
    return has_shard


def _load_shard(local: str, shard: int) -> bytes:
    """Read the specified dataset shard.

    Args:
        local (str): Local file path.
        shard (int): Shard ID.

    Returns:
        bytes: Shard data.
    """
    shard_filename = os.path.join(local, get_shard_basename(shard))
    return open(shard_filename, 'rb').read()

def _set_has_shard(local: str, shard: int, has: bool) -> None:
    """Set whether the specified shard has been downloaded.

    Args:
        local (str): Local file path.
        shard (int): Shard ID.
        has (bool): Whether downloaded.
    """
    has_shard_filename = os.path.join(local, get_has_shard_basename())
    with open(has_shard_filename, 'r+b', 0) as out:
        out.seek(shard)
        out.write(b'\1' if has else b'\0')


def _save_shard(shard_data: bytes, local: str, shard: int) -> None:
    """Save the given shard.

    Args:
        shard_data (bytes): Shard data.
        local (str): Local dataset directory.
        shard (int): Shard ID.
    """
    shard_filename = os.path.join(local, get_shard_basename(shard))
    if os.path.exists(shard_filename):
        return
    with open(shard_filename, 'xb') as out:
        out.write(shard_data)
    _set_has_shard(local, shard, True)


def _distribute_shard(world: World, local: str, shard: int) -> None:
    """Distribute the given downloaded shards from rank zero to all ranks.

    Args:
        world (World): World info.
        local (str): Local dataset directory.
        shard (int): Shard ID.
    """
    if world.is_leader:
        shard_data = _load_shard(local, shard)
        _send_broadcast(world, shard_data)
    elif world.is_ng_local_leader:
        shard_data = _recv_broadcast(world, world.leader)
        _save_shard(shard_data, local, shard)
    else:
        _recv_broadcast(world, world.leader)


def _distribute_shards(world: World, local: str, num_shards: int) -> None:
    """Distribute already-downloaded shards to all ranks.

    Args:
        world (World): World info.
        local (str): Local dataset directory.
        num_shards (int): Number of shards to expect.
    """
    if world.num_nodes == 1:
        _list_shards(local, num_shards)
        return

    shards_per_process = [torch.empty(num_shards, dtype=torch.uint8, device=world.device)
                          for _ in range(world.world_size)]
    has_shard = _mp_list_shards(world, local, num_shards)
    shards_this_process = torch.tensor(has_shard, device=world.device)
    dist.all_gather(shards_per_process, shards_this_process)
    processes_per_shard = torch.stack(shards_per_process, 1)
    for shard, processes in enumerate(processes_per_shard):
        if len(set(processes.cpu().numpy())) == 1:
            continue
        if not processes[0]:
            continue
        _distribute_shard(world, local, shard)


def _download_and_distribute_shard(world: World, remote: str, local: str, retry: int, timeout: float,
                                   shard: int) -> None:
    """Download and distribute the specified shard.

    Args:
        world (World): World info.
        remote (str): Remote dataset directory.
        local (str): Local dataset directory.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    if world.is_leader:
        remote_filename = os.path.join(remote, get_shard_basename(shard))
        local_filename = os.path.join(local, get_shard_basename(shard))
        _download(remote_filename, local_filename, retry, timeout)
        _set_has_shard(local, shard, True)

    if 1 < world.num_nodes:
        _distribute_shard(world, local, shard)


def _download_and_distribute_shards(world: World, remote: str, local: str, retry: int, timeout: float, seed: int,
                                    num_shards: int) -> None:
    """Download and distribute all missing shards, one at at time.

    Args:
        world (World): World info.
        remote (str): Remote dataset directory.
        local (str): Local dataset directory.
        retry (int): Number of retries.
        timeout (float): Connect and read timeout.
    """
    has_shard = _mp_list_shards(world, local, num_shards)
    missing_shards = np.argwhere(has_shard == 0).flatten()
    rng = np.random.default_rng(seed)
    rng.shuffle(missing_shards)
    for shard in missing_shards:
        _download_and_distribute_shard(world, remote, local, retry, timeout, shard)


class StreamingDataset(IterableDataset):
    def __init__(self, world: World, remote: str, local: str, shuffle: bool, decoders: Dict[str, Callable], retry: int,
                 timeout: float, seed: Optional[int]) -> None:
        self.world = world
        self.remote = remote
        self.local = local
        self.shuffle = shuffle
        self.decoders = decoders
        self.retry = retry
        self.timeout = timeout
        self.seed = _distribute_seed(world, seed)

        self.index = _download_and_distribute_index(world, remote, local, retry, timeout)
        _distribute_shards(world, local, self.index.num_shards)
        #Process(target=_download_and_distribute_shards,
        #        args=(world, remote, local, retry, timeout, self.seed, self.index.num_shards), daemon=True).start()
        _download_and_distribute_shards(world, remote, local, retry, timeout, self.seed, self.index.num_shards)

        self._lock: Lock
        self._has_worker_init = False
        self._next_epoch_shm: SharedMemory
        self._is_downloaded: bool
        self._is_id_downloaded: NDArray[np.uint8]
        self._epoch2is_id_mine: Dict[int, NDArray[np.uint8]]
        self._epoch2todo_ids: Dict[int, List[int]]

    def _shards_to_samples(self, shards: NDArray[np.int64]) -> NDArray[np.int64]:
        """Get the samples of the given shards according to the index.

        Args:
            shards (NDArray[np.int64]): The shards.

        Returns:
            NDArray[np.int64]: The samples of the shards.
        """
        ids = []
        for shard in shards:
            begin = self.index.shard_begins[shard]
            end = self.index.shard_ends[shard]
            ids += list(range(begin, end))
        return np.array(ids)

    def _add_shards(self, shards: NDArray[np.int64]) -> None:
        """Insert the samples of the given shards, updating any dynamic epochs.

        Args:
            NDArray[np.int64]: The shards.
        """
        with self._lock:
            new_ids = self._shards_to_samples(shards)
            self._is_id_downloaded[new_ids] = 1
            for epoch, is_id_mine in self._epoch2is_id_mine.items():
                new_todo_ids = new_ids[np.argwhere(is_id_mine[new_ids]).flatten()].tolist()
                todo_ids = self._epoch2todo_ids[epoch]
                todo_ids.reverse()
                todo_ids.extend(new_todo_ids)
                todo_ids.reverse()

    def _poll_for_shards(self) -> None:
        """Poll for new shards to be downloaded, inserting their samples."""
        has_shard_filename = os.path.join(self.local, get_has_shard_basename())
        old_has_shard = np.zeros(self.index.num_shards, np.uint8)
        while True:
            with self._lock:
                has_shard = np.fromfile(has_shard_filename, np.uint8)
                new_shards = np.argwhere(has_shard - old_has_shard).flatten()
                self._add_shards(new_shards)
                old_has_shard = has_shard
                self._is_downloaded = bool(has_shard.all())
                if self._is_downloaded:
                    break
            sleep(1.337)

    def _maybe_worker_init(self) -> None:
        """Worker setup method, run once on first `__iter__`."""
        if self._has_worker_init:
            return
        self._has_worker_init = True
        self._lock = Lock()
        with self._lock:
            next_epoch_shm_name = os.path.join(str(_get_root_python_pid()), 'next_epoch')
            try:
                self._next_epoch_shm = SharedMemory(next_epoch_shm_name, True, np.int64().nbytes)
                self._next_epoch_shm.buf[:] = b'\0'
            except:
                self._next_epoch_shm = SharedMemory(next_epoch_shm_name, False, np.int64().nbytes)
            self._is_id_downloaded = np.zeros(self.index.total_samples, np.uint8)
            self._is_downloaded = False
            self._epoch2is_id_mine = {}
            self._epoch2todo_ids = {}
        Thread(target=self._poll_for_shards, daemon=True).start()

    def _next_epoch(self) -> int:
        """Get the epoch, incrementing the counter.

        Returns:
            int: Epoch.
        """
        epoch = np.frombuffer(self._next_epoch_shm.buf, np.int64)[0]
        if self.world.is_local_leader:
            info = get_worker_info()
            if info is None or info.id == 0:
                sleep(0.42)
                self._next_epoch_shm.buf[:] = np.int64(epoch + 1).tobytes()
        return epoch

    def _epoch_init(self) -> int:
        """Epoch setup method, run on `__iter__`.

        Returns:
            int: Epoch.
        """
        with self._lock:
            epoch = self._next_epoch()
            epoch_seed = self.seed + epoch
            rng = np.random.default_rng(epoch_seed)

            all_ids = rng.permutation(self.index.total_samples)
            begin = len(all_ids) * self.world.rank // self.world.world_size
            end = len(all_ids) * (self.world.rank + 1) // self.world.world_size
            my_ids = all_ids[begin:end]

            is_id_mine = np.zeros(self.index.total_samples, np.uint8)
            is_id_mine[my_ids] = 1
            self._epoch2is_id_mine[epoch] = is_id_mine

            todo_ids = np.argwhere(is_id_mine * self._is_id_downloaded).flatten()
            if self.shuffle:
                np.random.shuffle(todo_ids)
            self._epoch2todo_ids[epoch] = todo_ids.tolist()
        return epoch

    def __getitem__(self, idx: int) -> Any:
        """Get the sample dict for the given sample ID.

        Args:
            idx (int): Sample ID.

        Returns:
            Dict[str, bytes]: Sample dict of keys to bytes.
        """
        shard = self.index.sample_shards[idx]
        offset = self.index.sample_shard_offsets[idx]
        size = self.index.bytes_per_sample[idx]

        shard_filename = os.path.join(self.local, get_shard_basename(shard))
        with open(shard_filename, 'rb', 0) as fp:
            fp.seek(offset)
            sample_data = fp.read(size)

        key2raw = bytes_to_sample_dict(sample_data, self.index.fields)

        sample = {}
        for key, decode in self.decoders.items():
            sample[key] = decode(key2raw[key])

        return sample

    def _dynamic_iter_ids(self, epoch: int) -> Iterator[int]:
        """Iterate over sample IDs (loading while iterating).

        Args:
            epoch (int): Which epoch (different partition per epoch).

        Returns:
            Iterator[int]: Iterator over sample IDs.
        """
        todo_ids = self._epoch2todo_ids[epoch]
        while True:
            with self._lock:
                if todo_ids:
                    yield todo_ids.pop()
                elif self._is_downloaded:
                    del self._epoch2todo_ids[epoch]
                    break
            sleep(0.1337)

    def _static_iter_ids(self, epoch: int) -> Iterator[int]:
        """Iterate over sample IDs (fully loaded).

        Args:
            epoch (int): Which epoch (different partition per epoch).

        Returns:
            Iterator[int]: Iterator over sample IDs.
        """
        epoch_seed = self.seed + epoch
        rng = np.random.default_rng(epoch_seed)
        ids = rng.permutation(self.index.total_samples)
        begin = len(ids) * self.world.rank // self.world.world_size
        end = len(ids) * (self.world.rank + 1) // self.world.world_size
        yield from ids[begin:end]

    def _iter_ids(self) -> Iterator[int]:
        """Iterate over sample IDs.

        Returns:
            Iterator[int]: Iterator over sample IDs.
        """
        self._maybe_worker_init()
        epoch = self._epoch_init()

        with self._lock:
            iter_ids = self._static_iter_ids if self._is_downloaded else self._dynamic_iter_ids

        yield from iter_ids(epoch)

    def __iter__(self) -> Iterator[Dict[str, bytes]]:
        """Iterate over samples.

        Returns:
            Iterator[Dict[str, bytes]]: Iterator over raw sample dicts.
        """
        for idx in self._iter_ids():
            yield self[idx]


class StreamingImageClassDataset(StreamingDataset):
    def decode_image(self, data: bytes) -> Image.Image:
        """Decode the sample image.

        Args:
            data (bytes): The raw bytes.

        Returns:
            Image: PIL image encoded by the bytes.
        """
        return Image.open(BytesIO(data)).convert('RGB')

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
                 max_retries: int = 2,
                 timeout: float = 60,
                 batch_size: Optional[int] = None) -> None:
        world = World(c_dist.get_global_rank(), c_dist.get_world_size(), c_dist.get_local_world_size())
        decoders = {
            'x': self.decode_image,
            'y': self.decode_class,
        }
        seed = None
        super().__init__(world, remote, local, shuffle, decoders, max_retries, timeout, seed)
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


def main(args):
    remote = 's3://mosaicml-internal-dataset-cifar10/mds/1/train/'
    local = '/tmp/mds-cache/mds-cifar10/train/'
    shuffle = True
    decoders = {}
    retry = 2
    timeout = 120
    seed = None
    world = World(args.rank, args.world_size, args.local_world_size)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    dist.init_process_group('nccl', rank=args.rank, world_size=args.world_size)

    dataset = StreamingDataset(world, remote, local, shuffle, decoders, retry, timeout, seed)
    del dataset

    dist.destroy_process_group()


if __name__ == '__main__':
    main(parse_args())
