from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from composer import Trainer
from composer.algorithms import BlurPool, ChannelsLast, CutMix, LabelSmoothing
from composer.models import MNIST_Classifier

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST("data", download=True, train=True, transform=transform)
eval_dataset = datasets.MNIST("data", download=True, train=False, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=128)
eval_dataloader = DataLoader(eval_dataset, batch_size=128)

trainer = Trainer(
    model=MNIST_Classifier(num_classes=10),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="2ep",
    algorithms=[
        BlurPool(replace_convs=True, replace_maxpools=True, blur_first=True),
        ChannelsLast(),
        CutMix(num_classes=10),
        LabelSmoothing(smoothing=0.1),
    ]
)
trainer.fit()
