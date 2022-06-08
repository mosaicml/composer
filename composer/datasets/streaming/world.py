# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import struct
from socket import socket
from threading import Thread
from time import sleep
from typing import List

import numpy as np
import torch
from torch import distributed as dist

__all__ = ["World", "send", "recv", "broadcast", "gather"]


class World(object):

    def __init__(self, rank: int, world_size: int, local_world_size: int, master_host: str):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % local_world_size
        self.local_world_size = local_world_size

        self.device = torch.device(f'cuda:{self.local_rank}')

        self.leader = 0
        self.local_leaders = list(range(0, self.world_size, self.local_world_size))
        self.ng_local_leaders = list(range(self.local_world_size, self.world_size, self.local_world_size))
        self.ng_all = list(range(1, self.world_size))

        self.is_leader = not self.rank
        self.is_local_leader = not self.local_rank
        self.is_ng_local_leader = self.is_local_leader and not self.is_leader

        self.node = self.rank // local_world_size
        self.num_nodes = self.world_size // local_world_size

        if self.is_leader:
            sock = socket()
            sock.bind(('', 0))
            master_port = sock.getsockname()[1]
            tensor = torch.tensor([master_port], dtype=torch.int32, device=self.device)
            dist.broadcast(tensor, self.rank)
            sock.listen(128)
            pairs = []
            for _ in range(self.world_size - 1):
                conn, _ = sock.accept()
                data = conn.recv(4)
                rank = struct.unpack('>i', data)[0]
                pairs.append((rank, conn))
            pairs.sort()
            self.socks = [pair[1] for pair in pairs]
        else:
            tensor = torch.empty(1, dtype=torch.int32, device=self.device)
            dist.broadcast(tensor, self.leader)
            master_port = int(tensor)
            sleep(3.1337)
            self.sock = socket()
            self.sock.connect((master_host, master_port))
            data = struct.pack('>i', self.rank)
            self.sock.send(data)


def send(sock: socket, data: bytes) -> None:
    """Write data to the socket.

    Args:
        sock (socket): Socket.
        data (bytes): Data.
    """
    metadata = struct.pack('>q', len(data))
    sock.sendall(metadata)
    sock.sendall(data)


def _recv_sized(sock: socket, size: int, chunk: int = 1024) -> bytes:
    """Read given number of bytes from the socket.

    Args:
        sock (socket): Socket.
        size (int): Bytes to read.
        chunk (int): Chunk size. Default: 1024.

    Returns:
        bytes: Data.
    """
    data = bytearray(np.zeros(size, np.uint8))
    begin = 0
    while begin < size:
        end = min(begin + chunk, size)
        part = sock.recv(end - begin)
        part_size = len(part)
        data[begin:begin + part_size] = part
        begin += part_size
    return bytes(data)


def recv(sock: socket, chunk: int = 1024) -> bytes:
    """Read data from the socket.

    Args:
        sock (socket): Socket.
        chunk (int): Chunk size. Default: 1024.

    Returns:
        bytes: Data.
    """
    metadata = _recv_sized(sock, 8, chunk)
    size = struct.unpack('>q', metadata)[0]
    return _recv_sized(sock, size, chunk)


def broadcast(socks: List[socket], ranks: List[int], data: bytes) -> None:
    """Broadcast data to the given ranks.

    Args:
        socks (List[socket]): Socket per rank.
        ranks (List[int]): Ranks to broadcast to.
        data (bytes): Data to broadcast.
    """
    threads = []
    for rank in ranks:
        t = Thread(target=send, args=(socks[rank - 1], data))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def _recv_append(sock: socket, ret: List[bytes], chunk: int = 1024) -> None:
    """Receive data, then append to list (used to return values from threads).

    Args:
        sock (socket): Socket.
        ret (List[bytes]): Return value container.
        chunk (int): Chunk size. Default: 1024.
    """
    ret.append(recv(sock, chunk))


def gather(socks: List[socket], ranks: List[int], chunk: int = 1024) -> List[bytes]:
    """Gather data from the given ranks.

    Args:
        socks (List[socket]): Socket per rank.
        ranks (List[int]): Ranks to gather from.
        chunk (int): Chunk size. Default: 1024.

    Returns:
        List[bytes]: Gathered data.
    """
    lists = [[] for _ in ranks]
    threads = []
    for idx, rank in enumerate(ranks):
        t = Thread(target=_recv_append, args=(socks[rank - 1], lists[idx], chunk))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return list(map(lambda items: items[0], lists))
