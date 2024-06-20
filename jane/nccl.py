import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run(rank, world_size):
    setup(rank, world_size)

    if rank == 0:
        print(f"Rank {rank}: Sleeping to simulate NCCL timeout")
        time.sleep(60)  # Sleep for 60 seconds to trigger timeout on other ranks
    else:
        try:
            print(f"Rank {rank}: Waiting for broadcast")
            tensor = torch.zeros(1).cuda(rank)
            dist.broadcast(tensor, src=0)
        except Exception as e:
            print(f"Rank {rank}: Caught exception - {e}")
            raise RuntimeError(f"Rank {rank}: NCCL timeout simulated exception") from e

    cleanup()


def main():
    world_size = 8
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
