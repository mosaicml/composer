import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

def test_redistribute_1gpu_to_2gpu():
    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Create a tensor on GPU
    rank = dist.get_rank()
    device = f"cuda:{rank}"
    
    # Create a tensor that will be distributed
    tensor = torch.ones(4, 4, device=device)
    
    # Create a device mesh with just GPU 0 (both ranks use the same mesh)
    mesh_1gpu = dist.DeviceMesh("cuda", [0])
    
    # Distribute the tensor to the 1 GPU mesh (replicated)
    if rank == 0:
        dtensor_1gpu = DTensor.from_local(tensor, mesh_1gpu)
    else:
        # create an empty DTensor
        dtensor_1gpu = DTensor.from_local(torch.empty(0), mesh_1gpu)
    # Create a device mesh with 2 GPUs
    mesh_2gpu = dist.DeviceMesh("cuda", [0, 1])
    
    # Redistribute the tensor to the 2 GPU mesh
    # We'll shard it along dimension 0
    dtensor_2gpu = dtensor_1gpu.redistribute(mesh_2gpu, [Shard(0)])
    
    # Print the local tensor on each rank
    print(f"Rank {rank} local tensor shape: {dtensor_2gpu.to_local().shape}")
    print(f"Rank {rank} local tensor: {dtensor_2gpu.to_local()}")
    
    # Verify the tensor was properly sharded
    if rank == 0:
        expected = torch.ones(2, 4, device=device)
        assert torch.allclose(dtensor_2gpu.to_local(), expected)
    else:
        expected = torch.ones(2, 4, device=device)
        assert torch.allclose(dtensor_2gpu.to_local(), expected)

if __name__ == "__main__":
    test_redistribute_1gpu_to_2gpu() 