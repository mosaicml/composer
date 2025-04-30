import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
import torch.distributed as dist
import argparse

from composer.utils.parallelism import FSDPConfig
from composer.distributed import prepare_fsdp_module

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


class WeightTiedModel(nn.Module):
    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features, out_features, bias=False)
        self.linear1._fsdp_wrap = True
        self.linear2._fsdp_wrap = True
        self.linear2.weight = self.linear1.weight

    def forward(self, x):
        # if dist.get_rank() == 0:
        #     print('before first linear1')
        #     print(x.shape)
        #     print('linear 1 weight:', self.linear1.weight)
        #     print('linear 2 weight:', self.linear2.weight)
        #     print(self.linear1.weight is self.linear2.weight)
        x = self.linear1(x)
        # if dist.get_rank() == 0:
        #     print('after first linear1')
        #     print(x.shape)
        #     print('linear 1 weight:', self.linear1.weight)
        #     print('linear 2 weight:', self.linear2.weight)
        #     print(self.linear1.weight is self.linear2.weight)
        x = self.linear2(x)
        return x


def compare_fsdp_model(use_orig_params=True):
    mesh = dist.init_device_mesh(device_type='cuda', mesh_shape=(2, ))
    rank = dist.get_rank()
    model = SimpleModel().to('cuda')
    print(f"before fsdp on rank: {rank}")
    for name, param in model.named_parameters():
        print(name, param)
    handle = dist.barrier(async_op=True)
    handle.wait()
    fsdp_model = FSDP(model, use_orig_params=use_orig_params, device_mesh=mesh)
    model_state_dict = get_model_state_dict(fsdp_model, submodules=None, options=StateDictOptions())
    print("model state dict: ")
    print(model_state_dict)
    dist.barrier()
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
    handle = dist.barrier(async_op=True)
    handle.wait(timeout=1000000)
    fully_shard(model)
    print(f"after fsdp2 on rank: {rank}")
    for name, param in model.named_parameters():
        print(name, param)
    ipt = torch.tensor([[1., 2.]], device='cuda')
    print(model(ipt))

def test_prepare_fsdp_module():
    mesh = dist.init_device_mesh(device_type='cuda', mesh_shape=(2, ))
    rank = dist.get_rank()
    model = WeightTiedModel().to('cuda')
    prepare_fsdp_module(model, None, FSDPConfig())
    # model.linear1 = FSDP(model.linear1, use_orig_params=True, device_mesh=mesh)
    # model.linear2 = FSDP(model.linear2, use_orig_params=True, device_mesh=mesh)
    if rank == 0:
        print(f"after fsdp on rank: {rank}")
        print(model.linear1.weight)
        print(model.linear2.weight)
        print(model.linear1.weight is model.linear2.weight)
    ipt = torch.tensor([[1., 2., 3., 4.]], device='cuda')
    print(model(ipt))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fsdp2', action='store_true', help='Use FSDP2')
    # parser.add_argument('--use-orig-params', action='store_true', help='Use original parameters for FSDP')
    # args = parser.parse_args()
    # if not args.fsdp2:
    #     compare_fsdp_model(use_orig_params=args.use_orig_params)
    # else:
    #     compare_fully_shard_model()

    test_prepare_fsdp_module()