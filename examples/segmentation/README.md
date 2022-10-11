# Semantic Segmentation Example

This example illustrates how to train a semantic segmentation model in composer

## DeepLabv3+ on ADE20k

The `train_deeplabv3_ade20k.py` script trains a DeepLabv3+ model with either a ResNet-50 or ResNet-101 on the ADE20k semantic segmentation benchmark.

Before running the script, ADE20k semantic segmentation benchmark must be downloaded from [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip), then unzip. The path to the dataset will be used as the first argument to the script.

### Example configurations

<!--pytest.mark.skip-->

```bash
# Single GPU/CPU depending on torch.cuda.is_available():
python train_deeplabv3_ade20k.py /path/to/ade20k

# Log experiments to Weights and Biases:
python train_deeplabv3_ade20k.py /path/to/ade20k --wandb_logger --wandb_entity my_username --wandb_project my_project --wandb_run_name my_run_name

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
