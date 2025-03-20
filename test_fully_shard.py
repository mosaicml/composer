import torch
from torch.distributed.fsdp import fully_shard
import torch.nn as nn
import argparse  # Add argparse import

# Define a simple feedforward neural network
class SimpleFFN(nn.Module):
    def __init__(self, input_size=2, hidden_size=2, output_size=2):
        super(SimpleFFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SharedParamFFN(nn.Module):
    def __init__(self, input_size=2, hidden_size=2, output_size=2):
        super(SharedParamFFN, self).__init__()
        shared_weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.fc1.weight = shared_weight
        self.fc2.weight = shared_weight

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def fsdp_and_print(model):
    # Wrap the model with FSDP
    fully_shard(model)

    for param in model.parameters():
        print(param)
    print("Forward hooks:", model._forward_hooks)
    print("Forward pre-hooks:", model._forward_pre_hooks)
    print("Backward hooks:", model._backward_hooks)
    print("Backward pre-hooks:", model._backward_pre_hooks)
    print(fully_shard.state(model))

    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    for param in optim.param_groups[0]['params']:
        print(param)

def cpu_init():
    model = SimpleFFN()
    fsdp_and_print(model)

def meta_init():
    # init with meta device
    with torch.device('meta'):
        model = SimpleFFN()
    fsdp_and_print(model)
    model.to_empty(device='cpu')
    print("After to_empty")
    for param in model.parameters():
        print(param)

def fsdp_submodules(meta=False):
    if meta:
        with torch.device('meta'):
            model = SharedParamFFN()
    else:
        model = SharedParamFFN()
    model.fc1 = fully_shard(model.fc1)
    model.fc2 = fully_shard(model.fc2)

    print(model.fc1.weight)
    print(model.fc2.weight)
    print("Are fc1 and fc2 parameters the same?", model.fc1.weight is model.fc2.weight)
    print("Do fc1 and fc2 parameters share the same data pointer?", 
          model.fc1.weight.data_ptr() == model.fc2.weight.data_ptr(), "data_ptr is: ", model.fc1.weight.data_ptr())
    print("after init")
    model.to_empty(device='cpu')
    print(model.fc1.weight)
    print(model.fc2.weight)
    print("Are fc1 and fc2 parameters the same?", model.fc1.weight is model.fc2.weight)
    print("Do fc1 and fc2 parameters share the same data pointer?", 
          model.fc1.weight.data_ptr() == model.fc2.weight.data_ptr())
    print("Do fc1 and fc2 parameters'local tensor share the same data pointer?", 
          model.fc1.weight.to_local().data_ptr() == model.fc2.weight.to_local().data_ptr())

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Test fully_shard with different initialization methods')
    parser.add_argument('--init', type=str, required=True, choices=['cpu', 'meta'],
                        help='Initialization method: cpu or meta (required)')
    parser.add_argument('--tie', action='store_true', help='Test tied parameters with FSDP')
    args = parser.parse_args()

    # Note: Ensure that the distributed environment is initialized before this step
    torch.distributed.init_process_group(backend='gloo')

    if torch.distributed.get_rank() == 0:
        if args.init == 'cpu':
            print("cpu_init")
            cpu_init()
        
        if args.init == 'meta':
            print("meta_init")
            meta_init()

        if args.tie:
            print("fsdp_submodules")
            fsdp_submodules(args.init == 'meta')

