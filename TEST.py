# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from shutil import rmtree

from icecream import ic
from streaming import MDSWriter, StreamingDataset
from torch.distributed._tensor import Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.utils.data import DataLoader

from composer.callbacks import MemoryMonitor
from composer.loggers import InMemoryLogger
from composer.trainer.trainer import Trainer
from composer.utils import FSDPConfig, ParallelismConfig, TPConfig, dist, reproducibility
from tests.common import RandomClassificationDataset, SimpleComposerMLP

##########################################################################################
# Parameters
##########################################################################################

size: int = 4
batch_size: int = 1
num_classes: int = 2
num_features: int = 6
seed: int = 42
device: str = 'cuda'
cleanup: bool = False
output_dir: str = 'tmp/'
tensor_parallel_degree: int = 2

reproducibility.seed_all(seed)

rank: int = dist.get_local_rank()

##########################################################################################
# Dataloader
##########################################################################################

pt_dataset = RandomClassificationDataset(
    shape=(num_features,),
    num_classes=num_classes,
    size=size,
    device='cpu',
)

# clean directory
rmtree(output_dir)

columns = {'X': 'ndarray', 'y': 'int64'}
with MDSWriter(out=output_dir, columns=columns) as out:
    for i in range(len(pt_dataset)):
        X, y = pt_dataset[i]
        out.write({'X': X.numpy(), 'y': y.numpy()})

sm_dataset = StreamingDataset(
    local=output_dir,
    replication=tensor_parallel_degree,
    batch_size=batch_size,
)
dataloader = DataLoader(sm_dataset, batch_size=batch_size)

ic(rank)
for i in range(sm_dataset.num_samples):
    ic(sm_dataset[i])

##########################################################################################
# Model
##########################################################################################

model = SimpleComposerMLP(num_features=num_features, device=device, num_classes=num_classes)

##########################################################################################
# Parallelism Config
##########################################################################################

fsdp_config = FSDPConfig(
    state_dict_type='full',
    sharding_strategy='SHARD_GRAD_OP',
    mixed_precision='full',
    use_orig_params=True,
)
layer_plan = {
    'fc1': ColwiseParallel(),
    'fc2': RowwiseParallel(output_layouts=Shard(0)),
}
tp_config = TPConfig(layer_plan=layer_plan, tensor_parallel_degree=tensor_parallel_degree)
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
