# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import struct
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from threading import Lock, Thread
from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import IterableDataset, get_worker_info

from composer.datasets.streaming.download import download
from composer.datasets.streaming.format import (StreamingDatasetIndex, bytes_to_sample_dict, get_index_basename,
                                                get_shard_basename)
from composer.datasets.streaming.world import World, broadcast, gather, recv, send
from composer.utils import dist as c_dist


class StreamingDataset(IterableDataset):

    def _get_seed(self, seed: Optional[int] = None) -> int:
        """Get the seed.

        Args:
            seed (Optional[int]): Optional seed.

        Returns:
            int: Seed (randomly generated if not provided).
        """
        if seed is None:
            seed = np.random.choice(1 << 31)
        return int(np.int32(seed))

    def _distribute_seed(self, seed: Optional[int] = None) -> int:
        """Initialize and distribute the same seed from rank zero to all processes.

        Args:
            seed (Optional[int]): Optional seed.

        Returns:
            int: Seed from rank zero.
        """
        if self.world.world_size == 1:
            ret = self._get_seed(seed)
        elif self.world.is_leader:
            ret = self._get_seed(seed)
            data = struct.pack('>i', ret)
            broadcast(self.world.socks, self.world.ng_all, data)
        else:
            data = recv(self.world.sock)
            ret = struct.unpack('>i', data)[0]
        return ret

    def _download_index(self) -> bytes:
        """Download a StreamingDatasetIndex file from a remote filesystem.

        Returns:
            bytes: Index data.
        """
        basename = get_index_basename()
        local_filename = os.path.join(self.local, basename)
        if self.remote:
            remote_filename = os.path.join(self.remote, basename)
            download(remote_filename, local_filename, self.retry, self.timeout)
        else:
            assert os.path.isfile(local_filename)
        return open(local_filename, 'rb').read()

    def _save_index(self, data: bytes) -> None:
        """Save serialized StreamingDatasetIndex to a local streaming dataset directory.

        Args:
            data (bytes): Bytes to write.
        """
        filename = os.path.join(self.local, get_index_basename())
        if os.path.isfile(filename):
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'xb') as out:
            out.write(data)

    def _download_and_distribute_index(self) -> StreamingDatasetIndex:
        """Download and distribute a streaming dataset index.

        Returns:
            StreamingDatasetIndex: The index.
        """
        if self.world.world_size == 1:
            data = self._download_index()
        elif self.world.is_leader:
            data = self._download_index()
            broadcast(self.world.socks, self.world.ng_all, data)
        elif self.world.is_ng_local_leader:
            data = recv(self.world.sock)
            self._save_index(data)
        else:
            data = recv(self.world.sock)
        return StreamingDatasetIndex.loads(data)

    def _list_shards(self) -> NDArray[np.uint8]:
        """Get which shards are present or missing.

        Returns:
            NDArray[np.uint8]: Whether downloaded (byte per file).
        """
        have = set(os.listdir(self.local))
        has_shard = np.empty(self.index.num_shards, np.uint8)
        for shard in range(self.index.num_shards):
            want = get_shard_basename(shard)
            has_shard[shard] = want in have
        return has_shard

    def _broadcast_list_shards(self) -> NDArray[np.uint8]:
        """Get which shards are present or missing in the leader process to all processes.

        Returns:
            NDArray[np.uint8]: Whether downloaded (byte per file).
        """
        if self.world.world_size == 1:
            has_shard = self._list_shards()
        elif self.world.is_leader:
            has_shard = self._list_shards()
            broadcast(self.world.socks, self.world.ng_all, has_shard.tobytes())
        else:
            data = recv(self.world.sock)
            has_shard = np.frombuffer(data, np.uint8)
        return has_shard

    def _load_shard(self, shard: int) -> bytes:
        """Read the specified dataset shard.

        Args:
            shard (int): Shard ID.

        Returns:
            bytes: Shard data.
        """
        shard_filename = os.path.join(self.local, get_shard_basename(shard))
        return open(shard_filename, 'rb').read()

    def _save_shard(self, data: bytes, shard: int) -> None:
        """Save the given shard.

        Args:
            data (bytes): Shard data.
            shard (int): Shard ID.
        """
        shard_filename = os.path.join(self.local, get_shard_basename(shard))
        if os.path.exists(shard_filename):
            return
        with open(shard_filename, 'xb') as out:
            out.write(data)

    def _note_shard(self, shard: int) -> None:
        """Note shard as downloaded.

        Only called by local leaders.

        Args:
            shard (int): Shard ID.
        """
        self.has_shard_shm.buf[shard] = 1
        has_shard = np.frombuffer(self.has_shard_shm.buf, np.uint8)
        self.has_all_shm.buf[0] = bool(has_shard.all())

    def _distribute_shard(self, shard: int) -> None:
        """Distribute the given downloaded shards from rank zero to all ranks.

        Only called by local leaders.

        Args:
            shard (int): Shard ID.
        """
        if self.world.num_nodes == 1:
            pass
        elif self.world.is_leader:
            data = self._load_shard(shard)
            broadcast(self.world.socks, self.world.ng_local_leaders, data)
        else:
            data = recv(self.world.sock)
            self._save_shard(data, shard)
        self._note_shard(shard)

    def _distribute_shards(self) -> None:
        """Distribute already-downloaded shards to all nodes.

        Only called by local leaders.

        Returns:
            NDArray[np.uint8]: Shard download statuses.
        """
        if self.world.is_leader:
            shards_this_node = self._list_shards()
            shards_per_other_node = gather(self.world.socks, self.world.ng_local_leaders)
            shards_per_node = [shards_this_node.tobytes()] + shards_per_other_node
            shards_per_node = np.stack(list(map(lambda data: np.frombuffer(data, np.uint8), shards_per_node)))
            broadcast(self.world.socks, self.world.ng_local_leaders, shards_per_node.tobytes())
        else:
            shards_this_node = self._list_shards()
            send(self.world.sock, shards_this_node.tobytes())
            data = recv(self.world.sock)
            shards_per_node = np.frombuffer(data, np.uint8).reshape(self.world.num_nodes, self.index.num_shards)

        nodes_per_shard = shards_per_node.transpose()
        for shard, nodes in enumerate(nodes_per_shard):
            if not nodes[0]:
                continue
            elif (nodes == nodes[0]).all():
                continue
            self._distribute_shard(shard)

    def _download_shard(self, shard: int) -> int:
        """Download the specified shard.

        Only called by the global leader.

        Args:
            shard (int): Shard ID.

        Returns:
            int: Shard ID (return the ID because these calls can complete out of order).
        """
        basename = get_shard_basename(shard)
        local_filename = os.path.join(self.local, basename)
        if self.remote:
            remote_filename = os.path.join(self.remote, basename)
            download(remote_filename, local_filename, self.retry, self.timeout)
        else:
            assert os.path.isfile(local_filename)
        return shard

    def _download_and_distribute_shards(self) -> None:
        """Download and distribute all missing shards using process pool imap_unordered.

        Only called by local leaders.
        """
        has_shard = self._broadcast_list_shards()
        missing_shards = np.argwhere(has_shard == 0).flatten()
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(missing_shards)
        pool = Pool()

        if self.world.num_nodes == 1:
            for shard in pool.imap_unordered(self._download_shard, missing_shards):
                self._note_shard(shard)
        elif self.world.is_leader:
            for shard in pool.imap_unordered(self._download_shard, missing_shards):
                data = self._load_shard(shard)
                metadata = struct.pack('>i', shard)
                broadcast(self.world.socks, self.world.ng_local_leaders, metadata)
                broadcast(self.world.socks, self.world.ng_local_leaders, data)
                self._note_shard(shard)
        else:
            for _ in missing_shards:
                metadata = recv(self.world.sock)
                shard = struct.unpack('>i', metadata)[0]
                data = recv(self.world.sock)
                self._save_shard(data, shard)
                self._note_shard(shard)

    def _get_shm(self, size: int) -> SharedMemory:
        """Initialize or attach shared memory of the given size.

        Args:
            size (int): Size in bytes.

        Returns:
            SharedMemory: The shared memory.
        """
        value = np.random.choice(1 << 63)
        name = f'{value:016x}'
        if self.world.is_leader:
            broadcast(self.world.socks, self.world.ng_all, name.encode('utf-8'))
        else:
            name = recv(self.world.sock).decode('utf-8')

        try:
            shm = SharedMemory(name, True, size)
        except:
            shm = SharedMemory(name, False, size)

        return shm

    def __init__(self,
                 remote: Optional[str],
                 local: str,
                 shuffle: bool,
                 decoders: Dict[str, Callable],
                 retry: int = 2,
                 timeout: float = 60,
                 seed: Optional[int] = None,
                 batch_size: Optional[int] = None) -> None:
        self.world = World(c_dist.get_global_rank(), c_dist.get_world_size(), c_dist.get_local_world_size(),
                           os.environ['MASTER_ADDR'])
        self.remote = remote
        self.local = local
        self.shuffle = shuffle
        self.decoders = decoders
        self.retry = retry
        self.timeout = timeout
        self.seed = self._distribute_seed(seed)
        self.next_epoch_shm = self._get_shm(np.int64().nbytes)
        self.index = self._download_and_distribute_index()
        self.has_shard_shm = self._get_shm(self.index.num_shards)
        self.has_all_shm = self._get_shm(1)
        if self.world.is_local_leader:
            self._distribute_shards()
            Thread(target=self._download_and_distribute_shards).start()

    def __len__(self) -> int:
        """Get per-device number of samples.

        Returns:
            int: Per-device dataset size.
        """
        return self.index.total_samples // self.world.world_size

    def is_downloaded(self) -> bool:
        """Get whether all shards have been downloaded."""
        return bool(self.has_all_shm.buf[0])

    def _wait_and_step_epoch(self) -> None:
        """Wait for all workers to get the current epoch, then increment it."""
        sleep(0.5)
        epoch = np.frombuffer(self.next_epoch_shm.buf, np.int64)
        self.next_epoch_shm.buf[:] = np.int64(epoch + 1).tobytes()

    def _step_epoch(self) -> int:
        """Get what epoch this is, incrementing the counter if we are the first worker.

        Returns:
            int: Epoch.
        """
        if self.world.is_local_leader:
            info = get_worker_info()
            if info is None or info.id == 0:
                Thread(target=self._wait_and_step_epoch, daemon=True).start()
        return int(np.frombuffer(self.next_epoch_shm.buf, np.int64))

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
        return np.array(ids, dtype=np.int64)

    def _add_shards(self, is_id_mine: NDArray[np.uint8], lock: Lock, todo_ids: List[int],
                    shards: NDArray[np.int64]) -> None:
        """Add shard samples.

        Args:
            my_ids (NDArray[np.int64]): Sample IDs of our partition across all shards.
            lock (Lock): Lock for modifying `todo_ids`.
            todo_ids (List[int]): List of downloaded our-partition sample IDs remaining to use this epoch.
        """
        ids = self._shards_to_samples(shards)
        my_ids = ids[np.argwhere(is_id_mine[ids]).flatten()]
        if self.shuffle:
            np.random.shuffle(my_ids)
        with lock:
            todo_ids.reverse()
            todo_ids.extend(my_ids)
            todo_ids.reverse()

    def _poll_for_shards(self, is_id_mine: NDArray[np.uint8], lock: Lock, todo_ids: List[int],
                         old_has_shard: NDArray[np.uint8]) -> None:
        """Poll shared memory for newly downloaded shards, adding our samples of those shards.

        Args:
            is_id_mine (NDArray[np.uint8]): Whether each sample ID is in our partition.
            lock (Lock): Lock for modifying `todo_ids`.
            todo_ids (List[int]): List of downloaded our-partition sample IDs remaining to use this epoch.
            old_has_shard (NDArray[np.uint8]): Whether each shard is already downloaded.
        """
        while True:
            has_shard = np.frombuffer(self.has_shard_shm.buf, np.uint8).copy()
            new_shards = np.argwhere(has_shard - old_has_shard).flatten()
            self._add_shards(is_id_mine, lock, todo_ids, new_shards)
            if has_shard.all():
                return
            old_has_shard = has_shard
            sleep(1)

    def _iter_ids(self) -> Iterator[int]:
        """Iterate over sample IDs.

        Returns:
            Iterator[int]: Iterator over sample IDs.
        """
        epoch = self._step_epoch()
        epoch_seed = self.seed + epoch
        rng = np.random.default_rng(epoch_seed)
        all_ids = rng.permutation(self.index.total_samples)

        rank = self.world.rank
        world_size = self.world.world_size
        info = get_worker_info()
        if info:
            rank = rank * info.num_workers + info.id
            world_size = world_size * info.num_workers
        begin = len(all_ids) * rank // world_size
        end = len(all_ids) * (rank + 1) // world_size
        my_ids = all_ids[begin:end]

        if self.is_downloaded():
            if self.shuffle:
                np.random.shuffle(my_ids)
            yield from my_ids
        else:
            is_id_mine = np.zeros(self.index.total_samples, np.uint8)
            is_id_mine[my_ids] = 1
            lock = Lock()
            todo_ids = []
            has_shard = np.frombuffer(self.has_shard_shm.buf, np.uint8).copy()
            new_shards = np.argwhere(has_shard).flatten()
            self._add_shards(is_id_mine, lock, todo_ids, new_shards)
            Thread(target=self._poll_for_shards, args=(is_id_mine, lock, todo_ids, has_shard)).start()

            while True:
                with lock:
                    if todo_ids:
                        yield todo_ids.pop()
                        continue
                    elif self.is_downloaded():
                        break
                sleep(1)

    def __getitem__(self, idx: int) -> Any:
        """Get the sample dict for the given sample ID.

        Args:
            idx (int): Sample ID.

        Returns:
            Any: Sample dict of keys to bytes.
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

    def __iter__(self) -> Iterator[Any]:
        """Iterate over samples.

        Returns:
            Iterator[Any]: Iterator over samples.
        """
        for idx in self._iter_ids():
            yield self[idx]
