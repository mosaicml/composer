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
from composer.loss import soft_cross_entropy
from composer.metrics import CrossEntropy
from composer.models.tasks import ComposerClassifier
from composer.optim import CosineAnnealingWithWarmupScheduler, DecoupledSGDW
from composer.utils import dist

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

parser = argparse.ArgumentParser()

# Dataloader arguments
parser.add_argument('--data_dir',
                    help='Path to the directory containing the ImageNet-1k dataset',
                    type=str,
                    required=True)
parser.add_argument('--train_crop_size', help='Training image crop size', type=int, default=224)
parser.add_argument('--eval_resize_size', help='Evaluation image resize size', type=int, default=256)
parser.add_argument('--eval_crop_size', help='Evaluation image crop size', type=int, default=224)
parser.add_argument('--train_batch_size', help='Train dataloader batch size', type=int, default=2048)
parser.add_argument('--val_batch_size', help='Validation dataloader batch size', type=int, default=2048)

# Model arguments
parser.add_argument('--model_name',
                    help='Name of the resnet model to train',
                    default='resnet50',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])

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
parser.add_argument('--checkpoint_dir',
                    help='Directory to store checkpoints',
                    type=str,
                    default='checkpoint/{run_name}')
parser.add_argument('--checkpoint_interval', help='Frequency to save checkpoints', type=str, default='1ep')

# TODO: Load checkpoint arguments

# Recipes
parser.add_argument('--recipe_name',
                    help='Either "mild", "medium" or "spicy" in order of increasing training time and accuracy',
                    type=str,
                    choices=['mild', 'medium', 'spicy'])

# Trainer arguments
parser.add_argument('--seed', help='Random seed', type=int, default=17)
parser.add_argument('--grad_accum', help='Gradient accumulation either an int or "auto"', default='auto')
parser.add_argument('--max_duration',
                    help='Duration to train specified in terms of Time',
                    type=Time.from_timestring,
                    default='90ep')
parser.add_argument('--device', help='Device to run training on', choices=['gpu', 'cpu', 'tpu', 'mps'], default='gpu')
parser.add_argument('--precision',
                    help='Numerical precision for training',
                    choices=['fp32', 'fp16', 'amp'],
                    default='amp')

# Local storage checkpointing
args = parser.parse_args()

IMAGENET_CHANNEL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_CHANNEL_STD = (0.229, 0.224, 0.225)

# Train dataset
print('Build train dataloader')
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(args.train_crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD)
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
    #    collate_fn=pil_image_collate,  # Converts PIL image lists to a torch.Tensor with the channel dimension first
    persistent_workers=True,  # Reduce overhead of creating new workers at the expense of using slightly more RAM
)
print('Built train dataloader')

# Validation dataset
print('Build evaluation dataloader')
eval_transforms = transforms.Compose([
    transforms.Resize(args.eval_resize_size),
    transforms.CenterCrop(args.eval_crop_size),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_CHANNEL_MEAN, IMAGENET_CHANNEL_STD)
])
eval_dataset = ImageFolder(os.path.join(args.data_dir, 'val'), eval_transforms)
# Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup,
eval_sampler = dist.get_sampler(train_dataset, drop_last=False, shuffle=False)
eval_dataloader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=args.val_batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=False,
    sampler=eval_sampler,
    #    collate_fn=pil_image_collate,  # Converts PIL image lists to a torch.Tensor with the channel dimension first
    persistent_workers=True,  # Reduce overhead of creating new workers at the expense of using slightly more RAM
)
print('Built evaluation dataloader')

# Instantiate torchvision ResNet model
model_fn = getattr(resnet, args.model_name)
model = model_fn(num_classes=1000, groups=1, width_per_group=64)


# Specify model initialization
def weight_init(w: torch.nn.Module):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)


model.apply(weight_init)

# Performance metrics to log outside of training loss
train_metrics = Accuracy()
val_metrics = MetricCollection([CrossEntropy(), Accuracy()])

# Cross entropy loss that can handle both index and one-hot targets
loss_fn = soft_cross_entropy

# Wrapper function to convert a classification PyTorch model into a Composer model
composer_model = ComposerClassifier(model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn)

# Optimizer
optimizer = DecoupledSGDW(composer_model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

# Learning rate scheduler: LR warmup for 8 epochs, then cosine decay for the rest of training
# TODO: I really don't like alpha_f
lr_scheduler = CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup, t_max=args.t_max, alpha_f=args.alpha_f)

# Callbacks for logging
speed_monitor = SpeedMonitor(window_size=50)
lr_monitor = LRMonitor()

# Callback for checkpointing
checkpoint_saver = CheckpointSaver(folder=args.checkpoint_dir, save_interval=args.checkpoint_interval)

# Recipes for training ResNet architectures on ImageNet in order of increasing training time and accuracy
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

# Create the Trainer!
trainer = Trainer(
    model=composer_model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    eval_interval='1ep',
    optimizers=optimizer,
    schedulers=lr_scheduler,
    algorithms=algorithms,
    max_duration=args.max_duration,
    callbacks=[speed_monitor, lr_monitor],
    device=args.device,
    precision=args.precision,  # super fast mixed precision training
    grad_accum=args.grad_accum,
    seed=args.seed)

# Start training!
trainer.fit()
