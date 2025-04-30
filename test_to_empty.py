from torch.distributed.tensor import DTensor
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
import torch

import torch.nn as nn

device_mesh = DeviceMesh("cuda", [0, 1])
with torch.device('meta'):
    linear1 = nn.Linear(10, 10)
    linear2 = nn.Linear(10, 10)
# linear1.weight = nn.Parameter(DTensor.from_local(linear1.weight, device_mesh=device_mesh, placements=[Shard(0)]))
linear2.weight = linear1.weight

print(linear1.weight is linear2.weight)

linear1.to_empty(device='cuda')

print(linear1.weight is linear2.weight)

print(torch.equal(linear1.weight, linear2.weight))