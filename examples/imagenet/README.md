# ImageNet Examples

These examples illustrate how to use Composer to train models on ImageNet-1k. The scripts contain training recipes for both strong baselines and high accuracy models, depending on the configuration specified by commandline arguments. Before running these examples, please pip install composer by running `pip install mosaicml`. The only prequisite is to have ImageNet already downloaded on the system used for training

## ResNet

The `train_resnet_imagenet1k.py` script is used to train models from the ResNet architecture family e.g. ResNet-50, ResNet-101. The models are pulled directly from torchvision and wrapped into a `ComposerModel`. This script contains several additional features such as saving checkpoints, resuming training from a saved checkpoint, and logging time-to-train, throughput, and learning rate throughout training. Below is a list of a few of the possible configurations:

```bash
# Single GPU/CPU depending on torch.cuda.is_available()
python train_resnet_imagenet1k.py /path/to/imagenet

# Log experiments to Weights and Biases
python train_resnet_imagenet1k.py /path/to/imagenet --wandb_logger --wandb_entity my_username --wandb_project my_project --wandb_run_name my_run_name

# Single/Multi GPU training (infers the number of GPUs available)
composer train_resnet_imagenet1k.py /path/to/imagenet

#Manually specify number of GPUs to use:
composer -n $N_GPUS train_resnet_imagenet1k.py /path/to/imagenet

# Mild ResNet recipe for fastest training to ~76.5% accuracy:
composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name mild --train_crop_size 176 --val_crop_size 224 --max_duration 36ep  --loss_name binary_cross_entropy

# Medium ResNet recipe highest accuracy with similar training time as baseline:
composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name medium --train_crop_size 176 --val_crop_size 224 --max_duration 135ep  --loss_name binary_cross_entropy

# Spicy ResNet recipe for our most accurate ResNet over a long training schedule:
composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name spicy --train_crop_size 176 --val_crop_size 224 --max_duration 270ep  --loss_name binary_cross_entropy
```