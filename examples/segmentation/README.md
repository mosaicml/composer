# Semantic Segmentation Example

This example illustrates how to train a semantic segmentation model in composer.

## Installation

First, install [Composer](https://github.com/mosaicml/composer) with `pip install mosaicml`. Additionally, our models are pulled from [MMsegmentation](https://github.com/open-mmlab/mmsegmentation), so follow the [MMcv install instructions](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) (which is dependent on your CUDA and PyTorch versions), then install MMsegmentation with `pip install mmsegmentation`.

Alternatively, we have publicly available Docker images to reproduce our results. Use `mosaicml/pytorch_vision:1.12.1_cu116-python3.9-ubuntu20.04` for running on GPUs or `mosaicml/pytorch_vision:1.12.1_cpu-python3.9-ubuntu20.04` for running on CPUs.

## DeepLabv3+ on ADE20k

The `train_deeplabv3_ade20k.py` script trains a DeepLabv3+ model with either a ResNet-50 or ResNet-101 backbone on the ADE20k semantic segmentation benchmark. To download ADE20k locally (~1 GB), specify the `--download` option when running the script, then the dataset will be downloaded data directory path i.e. the first argument.

We designed the script to be hackable, so try our recipes on your own models and datsets!
### Example configurations

<!--pytest.mark.skip-->

```bash
# Downloads ADE20k and does single GPU/CPU training depending on torch.cuda.is_available():
python train_deeplabv3_ade20k.py /path/to/ade20k --download

# Log experiments to Weights and Biases:
python train_deeplabv3_ade20k.py /path/to/ade20k --wandb_logger --wandb_entity my_username --wandb_project my_project --run_name my_run_name

# Single/Multi GPU training (infers the number of GPUs available):
composer train_deeplabv3_ade20k.py /path/to/ade20k

# Manually specify number of GPUs to use:
composer -n $N_GPUS train_deeplabv3_ade20k.py /path/to/ade20k

# Mild DeepLabv3+ recipe for fastest training to 45.6 mIoU:
composer train_deeplabv3_ade20k.py /path/to/ade20k/ --recipe_name mild --max_duration 25ep

# Medium DeepLabv3+ recipe for highest mIoU (49.15) with similar training time as baseline:
composer train_deeplabv3_ade20k.py /path/to/ade20k/ --recipe_name medium --max_duration 90ep

# Hot DeepLabv3+ recipe for highest mIoU (49.83) with a long training schedule:
composer train_deeplabv3_ade20k.py /path/to/ade20k --recipe_name hot --max_duration 256ep
```
