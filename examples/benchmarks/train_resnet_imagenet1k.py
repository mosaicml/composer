import argparse
import os

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet

from composer.datasets.utils import pil_image_collate
from composer.models.tasks import ComposerClassifier
from composer.optim import DecoupledSGDW
from composer.utils import dist

parser = argparse.ArgumentParser()

# Dataloader arguments
parser.add_argument('datadir', help='Path to the directory containing the ImageNet-1k dataset', type=str)
parser.add_argument('train_batch_size', help='Train dataloader batch size', type=int, default=2048)
parser.add_argument('val_batch_size', help='Validation dataloader batch size', type=int, default=2048)

# Composer model arguments
parser.add_argument('model_name',
                    help='Name of the resnet model to train',
                    default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])

args = parser.parse_args()

# Configurations

# Train / val transforms
train_crop_size = 224
val_resize_size = 256
val_crop_size = 224

IMAGENET_CHANNEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STD = (0.229, 0.224, 0.225)

# Train dataset
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(train_crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD)
])
train_dataset = ImageFolder(os.path.join(args.datadir, 'train'), train_transforms)
# Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup
train_sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    sampler=train_sampler,
    collate_fn=pil_image_collate,  # Converts PIL image lists to a torch.Tensor with the channel dimension first
    persistent_workers=True,  # Reduce overhead of creating new workers at the expense of using slightly more RAM
)

# Validation dataset
val_transforms = transforms.Compose([
    transforms.Resize(val_resize_size),
    transforms.CenterCrop(val_crop_size),
    transforms.Normalize(IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD)
])
val_dataset = ImageFolder(os.path.join(args.datadir, 'val'), val_transforms)
# Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup,
val_sampler = dist.get_sampler(train_dataset, drop_last=False, shuffle=False)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.val_batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
    sampler=val_sampler,
    collate_fn=pil_image_collate,  # Converts PIL image lists to a torch.Tensor with the channel dimension first
    persistent_workers=True,  # Reduce overhead of creating new workers at the expense of using slightly more RAM
)

# Create a Composer model
model_fn = getattr(resnet, args.model_name)
model = model_fn()

# Optimizer
optimizer = DecoupledSGDW()
