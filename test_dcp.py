from typing import Dict, Optional, Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from torch.distributed.fsdp import fully_shard, FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, get_model_state_dict, get_optimizer_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import random
import numpy as np
import os
import shutil
import argparse

CHECKPOINT_DIR: str = "checkpoint"


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    def __init__(self, features: int) -> None:
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(features, features, bias=False)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(features, features, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net2(self.relu(self.net1(x)))


def setup(world_size: int) -> DeviceMesh:
    mesh = dist.init_device_mesh(device_type='cuda', mesh_shape=(world_size, ))
    return mesh


def cleanup():
    dist.destroy_process_group()

def setup_model_and_optimizer(world_size: int, use_fully_shard: bool) -> tuple[nn.Module, torch.optim.Optimizer]:
    mesh = setup(world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel(2).to(mesh.device_type)
    if use_fully_shard:
        fully_shard(model, mesh=mesh)
    else:
        model = FSDP(model, device_mesh=mesh)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    return model, optimizer

def run_fsdp_checkpoint_save_example(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: str):
    optimizer.zero_grad()
    model(torch.rand(1, 2, device="cuda")).sum().backward()
    optimizer.step()

    state_dict = { "app": AppState(model, optimizer) }
    dcp.save(state_dict, checkpoint_id=checkpoint_dir)

def run_fsdp_checkpoint_load_example(model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_dir: str):
    state_dict = { "app": AppState(model, optimizer) }
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=checkpoint_dir,
    )
    optimizer.zero_grad()
    model(torch.rand(1, 2, device="cuda")).sum().backward()
    optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP Checkpoint Example")
    parser.add_argument("--fully-shard", action="store_true", help="Use fully_shard instead of FSDP")
    parser.add_argument("--load", action="store_true", help="Run the load example instead of the save example")
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, help="Directory to save/load checkpoints")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Purge all files/dirs in checkpoint_dir if running save example
    if not args.load and os.path.isdir(args.checkpoint_dir):
        shutil.rmtree(args.checkpoint_dir)

    model, optimizer = setup_model_and_optimizer(2, args.fully_shard)

    if args.load:
        run_fsdp_checkpoint_load_example(model, optimizer, args.checkpoint_dir)
    else:
        run_fsdp_checkpoint_save_example(model, optimizer, args.checkpoint_dir)
    cleanup()