import torch
from icecream import ic

from tests.trainer.test_tp import test_tp_forward
from tests.common import RandomClassificationDataset, SimpleComposerMLP


size: int = 4
batch_size: int = 1
num_classes: int = 2
num_features: int = 2
seed: int = 43
device: torch.device = torch.device('cuda')

dataset = RandomClassificationDataset(
    shape=(num_features,),
    num_classes=num_classes,
    size=size,
    device=device,
)

model = SimpleComposerMLP(num_features=num_features, device=device, num_classes=num_classes)

batch = dataset[0]
ic(batch)
output = model(batch)
ic(output)

