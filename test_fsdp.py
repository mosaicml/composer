import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
import argparse

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2, bias=False)
        self.linear1.weight.data = torch.tensor([[1., 2.], [3., 4.]])
        self.linear2 = nn.Linear(2, 2, bias=False)
        self.linear2.weight.data = torch.tensor([[5., 6.], [7., 8.]])

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    
def compare_fsdp_model(use_orig_params=True):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    model = SimpleModel().to('cuda')
    print(f"before fsdp on rank: {rank}")
    for name, param in model.named_parameters():
        print(name, param)
    fsdp_model = FSDP(model, device_id=0, use_orig_params=use_orig_params)
    print(f"after fsdp on rank: {rank}")
    for name, param in fsdp_model.named_parameters():
        print(name, param)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-orig-params', action='store_true', help='Use original parameters for FSDP')
    args = parser.parse_args()
    
    compare_fsdp_model(use_orig_params=args.use_orig_params)