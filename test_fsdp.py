import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import fully_shard
import torch.distributed as dist
import argparse

class SimpleModel(nn.Module):
    def __init__(self, in_features=2, out_features=2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear1.weight.data = torch.tensor([[1., 2.], [3., 4.]])
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.linear2.weight.data = torch.tensor([[5., 6.], [7., 8.]])

    def forward(self, x):
        print('linear 1 weight:', self.linear1.weight)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
def compare_fsdp_model(use_orig_params=True):
    dist.init_device_mesh(device_type='cuda', mesh_shape=(2, ))
    rank = dist.get_rank()
    model = SimpleModel().to('cuda')
    print(f"before fsdp on rank: {rank}")
    for name, param in model.named_parameters():
        print(name, param)
    dist.barrier()
    fsdp_model = FSDP(model, use_orig_params=use_orig_params)
    print(f"after fsdp on rank: {rank}")
    for name, param in fsdp_model.named_parameters():
        print(name, param)
    ipt = torch.tensor([[1., 2.]], device='cuda')
    print(fsdp_model(ipt))

def compare_fully_shard_model():
    dist.init_device_mesh(device_type='cuda', mesh_shape=(2, ))
    rank = dist.get_rank()
    model = SimpleModel().to('cuda')
    print(f"before fsdp2 on rank: {rank}")
    for name, param in model.named_parameters():
        print(name, param)
    dist.barrier()
    fully_shard(model)
    print(f"after fsdp2 on rank: {rank}")
    for name, param in model.named_parameters():
        print(name, param)
    ipt = torch.tensor([[1., 2.]], device='cuda')
    print(model(ipt))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsdp2', action='store_true', help='Use FSDP2')
    parser.add_argument('--use-orig-params', action='store_true', help='Use original parameters for FSDP')
    args = parser.parse_args()
    if not args.fsdp2:
        compare_fsdp_model(use_orig_params=args.use_orig_params)
    else:
        compare_fully_shard_model()