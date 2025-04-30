import os
import torch
import torch.distributed as dist
import argparse
import time
from datetime import timedelta


def init_distributed():
    """Initialize distributed environment with NCCL backend."""
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("Please run with torchrun or similar launcher")
        exit(1)
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize the distributed environment
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=5)
    )
    
    return rank, world_size

def test_barrier_timeout(timeout_seconds=5):
    """Test barrier timeout behavior."""
    rank, world_size = init_distributed()

    dist.barrier()
    
    print(f"Rank {rank}/{world_size} starting barrier test with timeout={timeout_seconds}s")
    
    try:
        group=dist.new_group(timeout=timedelta(seconds=10))

        # First, make rank 0 sleep before creating the barrier
        if rank == 0:
            print(f"Rank {rank} sleeping for {timeout_seconds + 2} seconds...")
            time.sleep(timeout_seconds + 2)

        # Now create the barrier after the sleep
        print(f"Rank {rank} creating barrier...")
        handle = dist.barrier(group=group)

        # Wait for the barrier with timeout
        print(f"Rank {rank} waiting for barrier...")
        # for i in range(10):
        #     print(f"Rank {rank} waiting for barrier... {i+1} seconds")
        #     time.sleep(1)
        #     if handle.is_completed():
        #         print(f"Rank {rank} barrier completed after {i+1} seconds")
        #         break
        # handle.wait()
        print(f"Rank {rank} passed barrier successfully")
        
    except RuntimeError as e:
        if "timed out" in str(e):
            print(f"Rank {rank} barrier timed out as expected")
        else:
            print(f"Rank {rank} encountered unexpected error: {e}")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=5.0,
                      help='Timeout in seconds for the barrier')
    args = parser.parse_args()
    
    test_barrier_timeout(args.timeout) 