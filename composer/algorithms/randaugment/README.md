# 🎲 RandAugment

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

`Computer Vision`

For each data sample, RandAugment randomly samples `depth` image augmentations from a set of augmentations (e.g. translation, shear, contrast) and applies them sequentially.
Each augmentation is applied with a context-specific `severity` sampled uniformly from 0 to 10.
Training in this fashion regularizes the network and can improve generalization performance.

| ![RandAugment](https://storage.googleapis.com/docs.mosaicml.com/images/methods/rand_augment.jpg) |
|:--:|
|*An image of a dog that undergoes three different augmentation chains. Each of these chains is a possible augmentation that might be applied by RandAugment and gets combined with the original image.*|

## How to Use

### Functional Interface

```python
# Run RandAugment on the image to produce a new RandAugmented image

from typing import Union

import torch
from PIL.Image import Image as PillowImage

import composer.functional as cf
from composer.algorithms.utils import augmentation_sets


def randaugment_image(image: Union[PillowImage, torch.Tensor]):
    randaugmented_image = cf.randaugment_image(img=image,
                                               severity=9,
                                               depth=2,
                                               augmentation_set=augmentation_sets["all"])
    return randaugmented_image
```

### Torchvision Transform

<!--pytest.mark.skip-->
```python
# Create a callable for RandAugment which can be composed with other image augmentations

import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset

from composer.algorithms.randaugment import RandAugmentTransform

randaugment_transform = RandAugmentTransform(severity=9,
                                             depth=2,
                                             augmentation_set="all")
composed = transforms.Compose([randaugment_transform, transforms.RandomHorizontalFlip()])
dataset = VisionDataset(data_path, transform=composed)
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomImageDataset, SimpleConvModel

model = SimpleConvModel()
train_dataloader = DataLoader(RandomImageDataset())
eval_dataloader = DataLoader(RandomImageDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate points in the training loop

from composer.algorithms import RandAugment
from composer.trainer import Trainer

randaugment_algorithm = RandAugment(severity=9,
                                    depth=2,
                                    augmentation_set="all")

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="1ep",
    algorithms=[randaugment_algorithm]
)

trainer.fit()
```

### Implementation Details

RandAugment leverages `torchvision.transforms` to add a transformation to the dataset which will be applied per image on the CPU. The transformation takes in a `PIL.Image` and outputs a `PIL.Image` with RandAugment applied.

The functional form of RandAugment (`randaugment_image()`) requires RandAugment hyperparameters when it is called.

The Torchvision transform form of RandAugment (`RandAugmentTransform`) is composable with other dataset transformations via `torchvision.transforms.Compose`.

The class form of RandAugment runs on `Event.FIT_START` and inserts `RandAugmentTransform` into the set of transforms in a `torchvision.datasets.VisionDataset` dataset.

## Suggested Hyperparameters

As per [Cubuk et al. (2020)](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html), we found that `depth=2` (applying a chain of two augmentations to each image) and `severity=9` (each augmentation is applied quite strongly) works well for different models of the ResNet family on ImageNet. For `depth≥3`, we find diminishing accuracy gains (due to over-regularization) and substantial training slowdown (due to the CPU becoming a bottleneck because of the amount of augmentation it must perform). We also recommend `augmentation_set=all` (using all available augmentation techniques).

> ❗ Potential CPU Bottleneck
>
> Further increasing `depth` beyond 2 significantly decreases throughput when training ResNet-50 on ImageNet due to bottlenecks in performing data augmentation on the CPU.

## Technical Details

RandAugment randomly samples `depth` image augmentations (with replacement) from the set of {`translate_x`, `translate_y`, `shear_x`, `shear_y`, `rotate`, `solarize`, `equalize`, `posterize`, `autocontrast`, `color`, `brightness`, `contrast`, `sharpness`}.
The augmentations use the PILLOW Image library (specifically Pillow-SIMD); we found OpenCV-based augmentations resulted in similar or worse performance.
RandAugment is applied after "standard" image transformations such as resizing and cropping, and before normalization.
Each augmentation is applied with an intensity randomly sampled uniformly from 0.1-`severity` (`severity` ≤ 10). where `severity` is a unit-free upper bound on the intensity of an augmentation and is mapped to the unit specific for each augmentation. For example, `severity` would be mapped to degrees for the rotation augmentation, and `severity=10` corresponds to 30°.

We observed an accuracy gain of 0.7% when adding RandAugment to a baseline ResNet-50 on ImageNet and 1.4% when adding RandAugment to a baseline ResNet-101 on ImageNet.
Cubuk et al. report a 1.3% accuracy increase over a baseline ResNet50 on ImageNet. They report identical accuracy as a previous state-of-the-art augmentation scheme (Fast AutoAugment) but with reduced computational requirements due to AutoAugment requiring a supplementary phase to search for the optimal augmentation policy.
However, the increased CPU load imposed by RandAugment in performing additional augmentations can also substantially reduce throughput: using RandAugment with the hyperparameters recommended by Cubuk et al. increased training time by up to 2.5x, depending on the hardware and model.

> ❗ Potential CPU Bottleneck
>
> We found that using RandAugment with the hyperparameters recommended by Cubuk et al. can increase the data augmentation load on the CPU so much that it bottlenecks training.
> Depending on the hardware configuration and model, we found that those hyperparameters increased training time by up to 2.5x.
>
> For example, applying RandAugment with `depth=2` when running on a high GPU to CPU resource system (1 Nvidia V100 — a relatively modern, powerful GPU — per 8 Intel Broadwell CPUs) causes throughput to decrease from ~612 im/sec/GPU to ~277 im/sec/GPU, while throughput remains at approximately at 212 im/sec/GPU on a low GPU to CPU system (1 Nvidia T4 — a relatively less powerful GPU — per 12 Intel Cascade Lake CPUs).

RandAugment will be more useful in overparameterized regimes (i.e. larger models) and for longer training runs.
Larger models typically take longer to run on a deep learning accelerator (e.g., a GPU), meaning there is more headroom to perform work on the CPU before augmentation becomes a bottleneck.
In addition, RandAugment is a regularization technique, meaning it reduces overfitting.
Doing so can allow models to reach higher quality, but this typically requires (1) larger models with more capacity to perform this more difficult learning and (2) longer training runs to allow these models time to learn.
For example, applying RandAugment with `depth=2`, `severity=9`, yields a 0.31% accuracy gain for ResNet-18 trained for 90 epochs, a 0.41% accuracy gain for ResNet-50 trained for 90 epochs, and a 1.15% gain for ResNet-50 trained for 120 epochs.

> 🚧 RandAugment May Reduce Quality for Smaller Models and Shorter Training Runs
>
> RandAugment is a regularization technique that makes training more difficult for the model.
> This can lead to higher model quality for longer training runs but may decrease accuracy for shorter training runs and require a larger model to overcome this difficulty.

> 🚧 Composing Regularization Methods
>
> As a general rule, composing regularization methods may lead to diminishing returns in quality improvements while increasing the risk of creating a CPU bottleneck.

> ❗ CIFAR-10C and ImageNet-C are no longer out-of-distribution
>
> [CIFAR-10C and ImageNet-C](https://github.com/hendrycks/robustness) are test sets created to evaluate the ability of models to generalize to images that are corrupted in various ways (i.e., images that are _out-of-distribution_ with respect to the standard CIFAR-10 and ImageNet training sets).
> These images were corrupted using some of the augmentation techniques in `augmentation_set=all`.
> If you use `augmentation_set=all`, these images are therefore no longer out-of-distribution.

## Attribution

[*Randaugment: Practical automated data augmentation with a reduced search space*](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html) by Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Published in CVPR 2020.

*The Composer implementation of this method and the accompanying documentation were produced by Matthew Leavitt at MosaicML.*
