from functools import partial
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributed._tensor.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from composer.trainer.trainer import Trainer
from composer.utils import dist
from composer.models import ComposerClassifier


class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape: Sequence[int] = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]
    

class SimpleModel(ComposerClassifier):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(
        self,
        num_features: int = 1,
        num_classes: int = 2,
        num_hidden: int = 8,
        device: str = 'cpu',
        bias: bool = True,
    ) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, num_hidden, device=device, bias=bias)
        fc2 = torch.nn.Linear(num_hidden, num_classes, device=device, bias=bias)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        net.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]
        super().__init__(module=net, num_classes=num_classes)

        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        # self.fc1 = fc1
        # self.fc2 = fc2

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


dist.initialize_dist('gpu')

model = SimpleModel()
dataset = RandomClassificationDataset(size=10)
dataloader = DataLoader(dataset, sampler=dist.get_sampler(dataset))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
state_dict_type = 'full'

layer_plan = {
    'module.2': ColwiseParallel(),
    'module.4': RowwiseParallel(),
}

tp_config = {
    'tensor_parallel_degree': 2,
    'layer_plan': layer_plan,
}

fsdp_config = {
    # 'data_parallel_shard_degree': 2,
    'state_dict_type': state_dict_type,
}

trainer = Trainer(
    model=model,
    optimizers=optimizer,
    train_dataloader=dataloader,
    tp_config={**tp_config},
    fsdp_config={**fsdp_config},
    progress_bar=False,
    log_to_console=True,
    max_duration='3ba',
    save_folder='./checkpoints',
    save_interval='1ba',
    save_overwrite=True,
)
trainer.fit()

state_dict = trainer.state.state_dict()
if state_dict_type == 'sharded' or dist.get_global_rank() == 0:
    print('\n\n[1, Saved]' + '*' * 50 + '\n')
    print(state_dict['model']['module.2.weight'])

model2 = SimpleModel()
trainer2 = Trainer(
    model=model2,
    optimizers=optimizer,
    train_dataloader=dataloader,
    tp_config={**tp_config},
    fsdp_config={**fsdp_config},
    progress_bar=False,
    log_to_console=True,
    max_duration='3ba',
    save_folder='./checkpoints',
    save_interval='1ba',
    save_overwrite=True,
    load_path='./checkpoints/ep0-ba3/',
    # load_path='./checkpoints/ep0-ba3-rank0.pt',
    # load_weights_only=True,
)

# print('\n\n[1.1, Random Init]' + '*' * 50 + '\n')
# print(trainer2.state.state_dict()['model']['module.2.weight'])

# from composer.utils import checkpoint
# checkpoint.load_checkpoint(path='./checkpoints/ep0-ba3/', state=trainer2.state, logger=trainer2.logger)

state_dict = trainer.state.state_dict()
if state_dict_type == 'sharded' or dist.get_global_rank() == 0:
    print('\n\n[3, Loaded]' + '*' * 50 + '\n')
    print(state_dict['model']['module.2.weight'])

# trainer2.fit()

