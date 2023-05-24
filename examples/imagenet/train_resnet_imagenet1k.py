# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to train a ResNet model on ImageNet."""

import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet

from composer import DataSpec, Time, Trainer
from composer.algorithms import (EMA, SAM, BlurPool, ChannelsLast, ColOut, LabelSmoothing, MixUp, ProgressiveResizing,
                                 RandAugment, StochasticDepth)
from composer.callbacks import CheckpointSaver, LRMonitor, SpeedMonitor
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.loggers import WandBLogger
from composer.loss import binary_cross_entropy_with_logits, soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models.tasks import ComposerClassifier
from composer.optim import CosineAnnealingWithWarmupScheduler, DecoupledSGDW
from composer.utils import dist

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

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
                    help='Duration of learning rate warmup specified as a Time string',
                    type=Time.from_timestring,
                    default='8ep')
parser.add_argument('--t_max',
                    help='Duration to cosine decay the learning rate specified as a Time string',
                    type=Time.from_timestring,
                    default='1dur')

# Save checkpoint arguments
parser.add_argument('--save_checkpoint_dir',
                    help='Directory in which to save model checkpoints',
                    type=str,
                    default='checkpoints/{run_name}')
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
parser.add_argument('--run_name', help='Name of the training run used for checkpointing and other logging', type=str)
parser.add_argument('--seed', help='Random seed', type=int, default=17)
parser.add_argument('--max_duration',
                    help='Duration to train specified as a Time string',
                    type=Time.from_timestring,
                    default='90ep')
parser.add_argument('--eval_interval',
                    help='How frequently to run evaluation on the validation set specified as a Time string',
                    type=Time.from_timestring,
                    default='1ep')

args = parser.parse_args()


def _main():

    # Divide batch sizes by number of devices if running multi-gpu training
    if dist.get_world_size():
        args.train_batch_size = args.train_batch_size // dist.get_world_size()
        args.eval_batch_size = args.eval_batch_size // dist.get_world_size()

    # Scale by 255 since the collate `pil_image_collate` results in images in range 0-255
    # If using ToTensor() and the default collate, remove the scaling by 255
    IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
    IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)

    # Train dataset
    logging.info('Building train dataloader')
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.train_crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
        transforms.RandomHorizontalFlip(),
    ])
    train_dataset = ImageFolder(os.path.join(args.data_dir, 'train'), train_transforms)
    # Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup
    train_sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)
    train_dataloader = DataLoader(
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
    logging.info('Built train dataloader\n')

    # Validation dataset
    logging.info('Building evaluation dataloader')
    eval_transforms = transforms.Compose([
        transforms.Resize(args.eval_resize_size),
        transforms.CenterCrop(args.eval_crop_size),
    ])
    eval_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), eval_transforms)
    # Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup,
    eval_sampler = dist.get_sampler(eval_dataset, drop_last=False, shuffle=False)
    eval_dataloader = DataLoader(
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
    logging.info('Built evaluation dataloader\n')

    # Instantiate torchvision ResNet model
    logging.info('Building Composer model')
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

    # Performance metrics to log other than training loss
    train_metrics = MulticlassAccuracy(num_classes=1000, average='micro')
    val_metrics = MetricCollection([CrossEntropy(), MulticlassAccuracy(num_classes=1000, average='micro')])

    # Cross entropy loss that can handle both index and one-hot targets

    if args.loss_name == 'binary_cross_entropy':
        loss_fn = binary_cross_entropy_with_logits
    else:
        loss_fn = soft_cross_entropy

    # Wrapper function to convert a classification PyTorch model into a Composer model
    composer_model = ComposerClassifier(model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn)
    logging.info('Built Composer model\n')

    # Optimizer
    logging.info('Building optimizer and learning rate scheduler')
    optimizer = DecoupledSGDW(composer_model.parameters(),
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

    # Learning rate scheduler: LR warmup for 8 epochs, then cosine decay for the rest of training
    lr_scheduler = CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup, t_max=args.t_max)
    logging.info('Built optimizer and learning rate scheduler\n')

    # Callbacks for logging
    logging.info('Building SpeedMonitor, LRMonitor, and CheckpointSaver callbacks')
    speed_monitor = SpeedMonitor(window_size=50)  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate

    # Callback for checkpointing
    checkpoint_saver = CheckpointSaver(folder=args.save_checkpoint_dir, save_interval=args.checkpoint_interval)
    logging.info('Built SpeedMonitor, LRMonitor, and CheckpointSaver callbacks\n')

    # Recipes for training ResNet architectures on ImageNet in order of increasing training time and accuracy
    # To learn about individual methods, check out "Methods Overview" in our documentation: https://docs.mosaicml.com/
    logging.info('Building algorithm recipes')
    if args.recipe_name == 'mild':
        algorithms = [
            BlurPool(),
            ChannelsLast(),
            EMA(half_life='100ba', update_interval='20ba'),
            ProgressiveResizing(initial_scale=0.5, delay_fraction=0.4, finetune_fraction=0.2),
            LabelSmoothing(smoothing=0.08),
        ]
    elif args.recipe_name == 'medium':
        algorithms = [
            BlurPool(),
            ChannelsLast(),
            EMA(half_life='100ba', update_interval='20ba'),
            ProgressiveResizing(initial_scale=0.5, delay_fraction=0.4, finetune_fraction=0.2),
            LabelSmoothing(smoothing=0.1),
            MixUp(alpha=0.2),
            SAM(rho=0.5, interval=10),
        ]
    elif args.recipe_name == 'spicy':
        algorithms = [
            BlurPool(),
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
    logging.info('Built algorithm recipes\n')

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
    logging.info('Building Trainer')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = 'amp' if device == 'gpu' else 'fp32'  # Mixed precision for fast training when using a GPU
    trainer = Trainer(run_name=args.run_name,
                      model=composer_model,
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
                      device=device,
                      precision=precision,
                      device_train_microbatch_size='auto',
                      seed=args.seed)
    logging.info('Built Trainer\n')

    # Start training!
    logging.info('Train!')
    trainer.fit()


if __name__ == '__main__':
    _main()
