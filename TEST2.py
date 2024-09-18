import torch
from icecream import ic
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import RowwiseParallel, ColwiseParallel
from torch.utils.data import DataLoader

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from tests.common import RandomClassificationDataset, SimpleComposerMLP
from composer.utils import FSDPConfig, ParallelismConfig, TPConfig, dist, reproducibility
from composer.devices import DeviceGPU


##########################################################################################
# Parameters
##########################################################################################

size: int = 4
batch_size: int = 1
num_classes: int = 2
num_features: int = 6
seed: int = 42
device: torch.device = 'cuda'


##########################################################################################
# Setup
##########################################################################################

reproducibility.seed_all(seed)

# if not dist.is_initialized():
#     ic('heyo')
#     dist.initialize_dist(DeviceGPU(), timeout=300.0)
# dist.barrier()


##########################################################################################
# DataLoader
##########################################################################################

dataset = RandomClassificationDataset(
    shape=(num_features,),
    num_classes=num_classes,
    size=size,
    device=device,
)  # X=(num_features,), y=(,), i.e. scalar
dataloader = DataLoader(
    dataset,
    sampler=dist.get_sampler(dataset),
    batch_size=batch_size,
)  # X=(batch_size, num_features), y=(batch_size,)


##########################################################################################
# Model
##########################################################################################


model = SimpleComposerMLP(num_features=num_features, device=device, num_classes=num_classes)


##########################################################################################
# Parallelism Config
##########################################################################################

class GatherColwiseParallel(ColwiseParallel):
    """ColwiseParallel layer that all-gathers the inputs first."""

    def __init__(
        self,
        *,
        use_local_output: bool = True,
    ):
        super().__init__()
        # Inputs over the TP dimension are sharded by device batches.
        self.input_layouts = (Shard(0),)
        # All-gather inputs so that each GPU now has the same input activations.
        self.desired_input_layouts = (Replicate(),)
        # self.output_layouts = (Shard(-1), )
        self.use_local_output = use_local_output

fsdp_config = FSDPConfig(
    state_dict_type='full',
    sharding_strategy='SHARD_GRAD_OP',
    mixed_precision='full',
    use_orig_params=True,
)
layer_plan = {
    'fc1': GatherColwiseParallel(),
    'fc2': RowwiseParallel(output_layouts=Shard(0)),
}

tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=2)
parallelism_config = ParallelismConfig(fsdp=fsdp_config, tp=tp_config)


##########################################################################################
# Trainer
##########################################################################################

trainer = Trainer(
    seed=seed,
    device='gpu',
    model=model,
    max_duration='1ep',
    train_dataloader=dataloader,
    precision='fp32',
    parallelism_config=parallelism_config,
    callbacks=[MemoryMonitor()],
    loggers=[InMemoryLogger()],
    progress_bar=False,
    log_to_console=False,
)
