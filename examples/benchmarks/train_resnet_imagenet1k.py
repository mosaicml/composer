# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""
This script demonstrates how to train torchvision ResNet architectures on ImageNet using Composer's Trainer. The
default settings replicate our baseline results. Below illustrates how some arguments can be set to replicate
our fastest ResNet recipe, our highest accuracy ResNet recipe, and several other settings. This script contains several
additional features such as saving checkpoints, resume training from a saved checkpoint, speed monitor to track
training time and throughput, and a learning rate monitor to track learning rate.

Single GPU training:
    python train_resnet_imagenet1k.py /path/to/imagenet

Log experiments to Weights and Biases:
    python train_resnet_imagenet1k.py /path/to/imagenet --wandb_logger --wandb_entity my_username
    --wandb_project my_project --wandb_run_name my_run_name

Single/Multi GPU training (infers the number of GPUs available):
    composer train_resnet_imagenet1k.py /path/to/imagenet

Manually specify number of GPUs to use:
    composer -n $N_GPUS train_resnet_imagenet1k.py /path/to/imagenet

Mild ResNet recipe for fastest training to ~76.5% accuracy:
    composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name mild --train_crop_size 176 --val_crop_size 224
    --max_duration 36ep  --loss_name binary_cross_entropy

Medium ResNet recipe highest accuracy with similar training time as baseline:
    composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name medium --train_crop_size 176 --val_crop_size 224
    --max_duration 135ep  --loss_name binary_cross_entropy

Spicy ResNet recipe for our most accurate ResNet over a long training schedule:
    composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name spicy --train_crop_size 176 --val_crop_size 224
    --max_duration 270ep  --loss_name binary_cross_entropy
"""

import argparse
import logging
import os

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet

from composer import DataSpec, Time, Trainer
from composer.algorithms import BlurPool, ChannelsLast, EMA, LabelSmoothing, ProgressiveResizing, MixUp, SAM, ColOut, RandAugment, StochasticDepth
from composer.callbacks import SpeedMonitor, LRMonitor, CheckpointSaver
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.loggers import WandBLogger
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models.tasks import ComposerClassifier
from composer.optim import CosineAnnealingWithWarmupScheduler, DecoupledSGDW
from composer.utils import dist

parser = argparse.ArgumentParser()

# Dataloader arguments
parser.add_argument('data_dir', help='Path to the directory containing the ImageNet-1k dataset', type=str)
parser.add_argument('--train_crop_size', help='Training image crop size', type=int, default=224)
parser.add_argument('--eval_resize_size', help='Evaluation image resize size', type=int, default=256)
parser.add_argument('--eval_crop_size', help='Evaluation image crop size', type=int, default=224)
parser.add_argument('--train_batch_size', help='Train dataloader per-device batch size', type=int, default=2048)
parser.add_argument('--eval_batch_size', help='Validation dataloader per-device batch size', type=int, default=2048)

# Model arguments
parser.add_argument('--model_name',
                    help='Name of the resnet model to train',
                    default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--loss_name',
                    help='Name of the loss function to use for training',
                    default='cross_entropy',
                    choices=['cross_entropy', 'binary_cross_entropy'])

# Optimizer arguments
parser.add_argument('--learning_rate', help='Optimizer learning rate', type=float, default=2.048)
parser.add_argument('--momentum', help='Optimizer momentum', type=float, default=0.875)
parser.add_argument('--weight_decay', help='Optimizer weight decay', type=float, default=5.0e-4)

# LR scheduler arguments
parser.add_argument('--t_warmup',
                    help='Duration of learning rate warmup specified by a Time string',
                    type=Time.from_timestring,
                    default='8ep')
parser.add_argument('--t_max',
                    help='Duration to cosine decay the learning rate',
                    type=Time.from_timestring,
                    default='1dur')
parser.add_argument('--alpha_f', help='Learning rate multiplier to decay to', type=Time.from_timestring, default=0)

# Save checkpoint arguments
parser.add_argument('--save_checkpoint_dir',
                    help='Directory to save checkpoints',
                    type=str,
                    default='checkpoint/{run_name}')
parser.add_argument('--checkpoint_interval', help='Frequency to save checkpoints', type=str, default='1ep')

# Load checkpoint arguments, assumes resuming the previous training run instead of fine-tuning
parser.add_argument('--load_checkpoint_path', help='Path to the checkpoint to load', type=str)

# Recipes
parser.add_argument('--recipe_name',
                    help='Either "mild", "medium" or "spicy" in order of increasing training time and accuracy',
                    type=str,
                    choices=['mild', 'medium', 'spicy'])

# Logger parameters: progress bar logging is used by default
# Only has Weights and Biases option to reduce the number of arguments. Other loggers can be substituted in the script
parser.add_argument('--wandb_logger', help='Whether or not to log results to Weights and Biases', action='store_true')
parser.add_argument('--wandb_entity', help='WandB entity name', type=str)
parser.add_argument('--wandb_project', help='WandB project name', type=str)
parser.add_argument('--wandb_run_name', help='WandB run name', type=str)

# Trainer arguments
parser.add_argument('--seed', help='Random seed', type=int, default=17)
parser.add_argument('--grad_accum', help='Gradient accumulation either an int or "auto"', default='auto')
parser.add_argument('--max_duration',
                    help='Duration to train specified in terms of Time',
                    type=Time.from_timestring,
                    default='90ep')
parser.add_argument('--eval_interval',
                    help='How frequently to run the evaluation datasets',
                    type=Time.from_timestring,
                    default='1ep')
parser.add_argument('--device', help='Device to run training on', choices=['gpu', 'cpu', 'tpu', 'mps'], default='gpu')
parser.add_argument('--precision',
                    help='Numerical precision for training',
                    choices=['fp32', 'fp16', 'amp'],
                    default='amp')

# Local storage checkpointing
args = parser.parse_args()

# Divide batch sizes by number of devices if running multi-gpu training
if dist.get_world_size():
    args.train_batch_size = args.train_batch_size // dist.get_world_size()
    args.eval_batch_size = args.eval_batch_size // dist.get_world_size()

IMAGENET_CHANNEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STD = (0.229, 0.224, 0.225)

# Train dataset
logging.info('Build train dataloader')
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(args.train_crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
])
train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), train_transforms)
# Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup
train_sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
    sampler=train_sampler,
    collate_fn=pil_image_collate,
    persistent_workers=True,  # Reduce overhead of creating new workers at the expense of using slightly more RAM
)
# DataSpec allows for on-gpu transformations, marginally relieving dataloader bottleneck
train_dataspec = DataSpec(dataloader=train_dataloader,
                          device_transforms=NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD))
logging.info('Built train dataloader')

# Validation dataset
logging.info('Build evaluation dataloader')
eval_transforms = transforms.Compose([
    transforms.Resize(args.eval_resize_size),
    transforms.CenterCrop(args.eval_crop_size),
])
eval_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), eval_transforms)
# Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup,
eval_sampler = dist.get_sampler(train_dataset, drop_last=False, shuffle=False)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=args.eval_batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
    sampler=eval_sampler,
    collate_fn=pil_image_collate,
    persistent_workers=True,  # Reduce overhead of creating new workers at the expense of using slightly more RAM
)
eval_dataspec = DataSpec(dataloader=eval_dataloader,
                         device_transforms=NormalizationFn(mean=IMAGENET_CHANNEL_MEAN, std=IMAGENET_CHANNEL_STD))
logging.info('Built evaluation dataloader')

# Instantiate torchvision ResNet model
logging.info('Build Composer model')
model_fn = getattr(resnet, args.model_name)
model = model_fn(num_classes=1000, groups=1, width_per_group=64)


# Specify model initialization
def weight_init(w: torch.nn.Module):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)
    # When using binary cross entropy, set the classification layer bias to -log(num_classes)
    # to ensure the initial probabilities are approximately 1 / num_classes
    if args.loss_name == 'binary_cross_entropy' and isinstance(w, torch.nn.Linear):
        w.bias.data = torch.ones(w.bias.shape) * -torch.log(torch.tensor(w.bias.shape[0]))


model.apply(weight_init)

# Performance metrics to log outside of training loss
train_metrics = Accuracy()
val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

# Cross entropy loss that can handle both index and one-hot targets

if args.loss_name == 'binary_cross_entropy':
    loss_fn = binary_cross_entropy_with_logits
else:
    loss_fn = soft_cross_entropy

# Wrapper function to convert a classification PyTorch model into a Composer model
composer_model = ComposerClassifier(model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn)
logging.info('Built Composer model')

# Optimizer
logging.info('Build optimizer and learning rate scheduler')
optimizer = DecoupledSGDW(composer_model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

# Learning rate scheduler: LR warmup for 8 epochs, then cosine decay for the rest of training
lr_scheduler = CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup, t_max=args.t_max, alpha_f=args.alpha_f)
logging.info('Build optimizer and learning rate scheduler')

# Callbacks for logging
logging.info('Build SpeedMonitor, LRMonitor, and CheckpointSaver callbacks')
speed_monitor = SpeedMonitor(window_size=50)
lr_monitor = LRMonitor()

# Callback for checkpointing
checkpoint_saver = CheckpointSaver(folder=args.save_checkpoint_dir, save_interval=args.checkpoint_interval)
logging.info('Build SpeedMonitor, LRMonitor, and CheckpointSaver callbacks')

# Recipes for training ResNet architectures on ImageNet in order of increasing training time and accuracy
logging.info('Build algorithm recipes')
if args.recipe_name == 'mild':
    algorithms = [
        BlurPool(),  # Add anti-aliasing filters to strided convs and maxpools
        ChannelsLast(),
        EMA(half_life='100ba', update_interval='20ba'),
        ProgressiveResizing(initial_scale=0.5, delay_fraction=0.4, finetune_fraction=0.2),
        LabelSmoothing(smoothing=0.08),
    ]
elif args.recipe_name == 'medium':
    algorithms = [
        BlurPool(),  # Add anti-aliasing filters to strided convs and maxpools
        ChannelsLast(),
        EMA(half_life='100ba', update_interval='20ba'),
        ProgressiveResizing(initial_scale=0.5, delay_fraction=0.4, finetune_fraction=0.2),
        LabelSmoothing(smoothing=0.1),
        MixUp(alpha=0.2),
        SAM(rho=0.5, interval=10),
    ]
elif args.recipe_name == 'spicy':
    algorithms = [
        BlurPool(),  # Add anti-aliasing filters to strided convs and maxpools
        ChannelsLast(),
        EMA(half_life='100ba', update_interval='20ba'),
        ProgressiveResizing(initial_scale=0.6, delay_fraction=0.2, finetune_fraction=0.2),
        LabelSmoothing(smoothing=0.13),
        MixUp(alpha=0.25),
        SAM(rho=0.5, interval=5),
        ColOut(p_col=0.05, p_row=0.05),
        RandAugment(depth=1, severity=9),
        StochasticDepth(target_layer_name='ResNetBottleneck',
                        stochastic_method='sample',
                        drop_distribution='linear',
                        drop_rate=0.1)
    ]
else:
    algorithms = None
logging.info('Built algorithm recipes')

logger = None
if args.wandb_logger:
    if args.wandb_entity is None:
        raise ValueError('Please specify --wandb_entity argument')
    if args.wandb_project is None:
        raise ValueError('Please specify --wandb_project argument')
    if args.wandb_run_name is None:
        raise ValueError('Please specify --wandb_run_name argument')
    logger = WandBLogger(entity=args.wandb_entity, project=args.wandb_project, name=args.wandb_run_name)

# Create the Trainer!
logging.info('Build Trainer')
trainer = Trainer(model=composer_model,
                  train_dataloader=train_dataspec,
                  eval_dataloader=eval_dataspec,
                  eval_interval=args.eval_interval,
                  optimizers=optimizer,
                  schedulers=lr_scheduler,
                  algorithms=algorithms,
                  loggers=logger,
                  max_duration=args.max_duration,
                  callbacks=[speed_monitor, lr_monitor, checkpoint_saver],
                  load_path=args.load_checkpoint_path,
                  device=args.device,
                  precision=args.precision,
                  grad_accum=8,
                  seed=args.seed)
logging.info('Built Trainer')

# Start training!
logging.info('Train!')
trainer.fit()
