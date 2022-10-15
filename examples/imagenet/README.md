# ImageNet Examples

These examples illustrate how to train models on ImageNet-1k using Composer.

Prerequisites:
- Install Composer: `pip install mosaicml`
- ImageNet already downloaded on the system used for training

## ResNet

The `train_resnet_imagenet1k.py` script trains models from the ResNet architecture family e.g. ResNet-50, ResNet-101.

Few details about the script:
- Models are pulled directly from torchvision, then wrapped into a `ComposerModel`
- Saves checkpoints to "checkpoints/{run_name}" by default where `run_name` is random if not specified
- Logs time-to-train, throughput, and learning rate throughout training.

### Example configurations:

<!--pytest.mark.skip-->

```bash
# Single GPU/CPU depending on torch.cuda.is_available()
python train_resnet_imagenet1k.py /path/to/imagenet

# Log experiments to Weights and Biases
python train_resnet_imagenet1k.py /path/to/imagenet --wandb_logger --wandb_entity my_username --wandb_project my_project --wandb_run_name my_run_name

# Single/Multi GPU training (infers the number of GPUs available)
composer train_resnet_imagenet1k.py /path/to/imagenet

# Manually specify number of GPUs to use:
composer -n $N_GPUS train_resnet_imagenet1k.py /path/to/imagenet

# Mild ResNet recipe for fastest training to ~76.5% accuracy:
composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name mild --train_crop_size 176 --eval_crop_size 224 --max_duration 36ep  --loss_name binary_cross_entropy

# Medium ResNet recipe highest accuracy with similar training time as baseline:
composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name medium --train_crop_size 176 --eval_crop_size 224 --max_duration 135ep  --loss_name binary_cross_entropy

# Spicy ResNet recipe for our most accurate ResNet over a long training schedule:
composer train_resnet_imagenet1k.py /path/to/imagenet --recipe_name spicy --train_crop_size 176 --eval_crop_size 224 --max_duration 270ep  --loss_name binary_cross_entropy
```
