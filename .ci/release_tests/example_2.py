import torch

# adaptive_avg_pool2d_backward_cuda in mnist_classifier is not deterministic
torch.use_deterministic_algorithms(False)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import ChannelsLast, CutMix, LabelSmoothing
from composer.models import mnist_model

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST("data", download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST("data", download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)
eval_dataloader = DataLoader(eval_dataset, batch_size=128)

trainer = Trainer(
    model=mnist_model(),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="2ep",
    algorithms=[
        ChannelsLast(),
        CutMix(alpha=1.0),
        LabelSmoothing(smoothing=0.1),
    ]
)
trainer.fit()
