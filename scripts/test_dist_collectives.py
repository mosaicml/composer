import torch
import torch.distributed as dist
from composer.utils import dist as cdist
from composer.utils import get_device


from composer.utils import get_device; cdist.initialize_dist(get_device(None))
world_size = dist.get_world_size()
rank = dist.get_rank()


print('world size', world_size)
print('starting all gather')
tensor_in = torch.full((10,), rank, dtype=torch.int32).cuda()
tensor_out = torch.zeros(world_size, 10, dtype=torch.int32).cuda()
dist.all_gather_into_tensor(tensor_out, tensor_in)
print('all gathered tensor successfully!')
print(tensor_out)