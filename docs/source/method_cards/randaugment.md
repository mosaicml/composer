# 🎲 RandAugment

![rand_augment.jpg](https://storage.googleapis.com/docs.mosaicml.com/images/methods/rand_augment.jpg)

Three example realizations of RandAugment. From [RandAugment: Practical Automated Data Augmentation with a Reduced Search Space](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html) by Cubul et al. 2020

Tags: `Vision`, `Increased Accuracy`, `Increased CPU Usage`, `Method`, `Augmentation`, `Regularization`

## TL;DR

For each data sample, RandAugment randomly samples `depth` image augmentations from a set of augmentations (e.g. translation, shear, contrast) and applies them sequentially with intensity sampled uniformly from 0.1- `severity` **(`severity` ≤ 10) for each augmentation.

## Attribution

*[RandAugment: Practical Automated Data Augmentation with a Reduced Search Space](https://openaccess.thecvf.com/content_CVPRW_2020/html/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.html)* by Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Released as a CVPR workshop paper in 2020.

## Applicable Settings

RandAugment uses image augmentations, and hence it is only applicable to vision tasks. Cubuk et al. show in their paper introducing the method that it improves performance in image classification and segmentation tasks.

## Hyperparameters

- `depth` - The number of augmentations applied to each image. Equivalent to the quantity *n* as described below.
- `severity` - The maximum possible intensity of each augmentation.
- `augmentation_set` - The set of augmentations to sample from. `all` is the set  {`translate_x`, `translate_y`, `shear_x`, `shear_y`, `rotate`, `solarize`, `posterize`, `equalize`, `autocontrast`, `color`, `contrast`, `brightness`, `sharpness`}. `safe` excludes `color`, `contrast`, `brightness`, and `sharpness`, which are used to generate the CIFAR-10-C and ImageNet-C benchmark datasets for naturalistic robustness ([Hendrycks et al., 2019](https://arxiv.org/abs/1903.12261)). `original` uses implementations of `color`, `contrast`, `brightness`, and `sharpness` that replicate [existing](https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py) [implementations](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py).

<!--
[comment]: #  See TODO LINK TO LINE IN CODE for more details.
-->

## Example Effects

We observe an accuracy gain of 0.7% when adding RandAugment to a baseline ResNet-50 on ImageNet, and 1.4% when adding RandAugment to a baseline ResNet-101 on ImageNet. However, the increased CPU load imposed by RandAugment in performing additional augmentations can also substantially reduce throughput: using RandAugment with the hyperparameters recommended by Cubuk et al. can increase training time by up to 2.5x, depending on the hardware and model.

Cubuk et al. report a 1.3% accuracy increase over a baseline ResNet50 on ImageNet. They report identical accuracy as a previous state-of-the-art augmentation scheme ([Fast AutoAugment](https://arxiv.org/abs/1905.00397)), but with reduced computational requirements due to AutoAugment requiring a supplementary phase to search for the optimal augmentation policy.

## Implementation Details

As per Cubuk et al., RandAugment randomly samples *n* image augmentations (with replacement) from the set of {`translate_x`, `translate_y`, `shear_x`, `shear_y`, `rotate`, `solarize`, `posterize`, `equalize`, `autocontrast`, `color`, `contrast`, `brightness`, `sharpness`}. The augmentations use the [PILLOW Image library](https://pillow.readthedocs.io/en/stable/reference/Image.html). RandAugment is applied after "standard" transformations such as resizing and cropping, and before normalization. Each augmentation is applied with an intensity randomly sampled between 0.1 and *m*, where *m* is a unit-free upper bound on the intensity of an augmentation and is mapped to the unit specific for each augmentation. For example, *m* would be mapped to degrees for the rotation augmentation, and *m* = 10 corresponds to 30°.

## Suggested Hyperparameters

As per Cubuk et al., we found that `depth` of 2 and `severity` of 9 worked well for
different models of the ResNet family on ImageNet. We found diminishing accuracy gains and
substantial training slowdown for `depth` ≥ 3. We also recommend `augmentation_set` =
`all`.

## Considerations

We found that RandAugment can significantly decrease throughput. This is due to the increased CPU load from performing the image augmentations. We found that training time could increase by up to 2.5x when `depth` **= 2, however the magnitude of the slowdown is determined by the ratio of GPU to CPU resources. For example, applying RandAugment with `depth` = 2 when running on a high GPU to CPU resource system (1 Nvidia V100—a relatively modern, powerful GPU—per 8 Intel Broadwell CPUs) causes throughput to decrease from ~612 im/sec/GPU to ~277 im/sec/GPU, while throughput remains at approximately at 212 im/sec/GPU on a low GPU to CPU system (1 Nvidia T4—a relatively less powerful GPU—per 12 Intel Cascade Lake CPUs).

The regularization benefits of RandAugment also tend to yield a greater benefit in overparameterized regimes (i.e., larger models, smaller datasets, and longer training times). For example, applying RandAugment with `depth` = 2, `severity` = 9, yields a 0.31% accuracy gain for ResNet-18 trained for 90 epochs, a 0.41% accuracy gain for ResNet-50 trained for 90 epochs, and a 1.15% gain for ResNet-50 trained for 120 epochs.

..

    TOOD add in Resnet 101 into the line above


```{eval-rst}
.. csv-table:: RandAugment
        :header: "Model","Baseline","RandAugment","Absolute Improvement (pp)","Relative Improvement"

        "ResNet18",70.19,70.4,0.21,1.003
        "ResNet50",76.19,76.61,0.42,1.006
        "ResNet50 (119 epochs)",75.78,76.93,1.15,1.015
        "ResNet101",77.18,78.19,1.01,1.013
```

RandAugment is also more suitable for larger models because large model workloads are more likely to be GPU-bound, leaving spare CPU capacity to perform the augmentations without becoming a bottleneck. RandAugment will be much more likely to impose a bottleneck on a small model for which each step is relatively faster on a GPU.

## Composability

As general rule, combining regularization-based methods yields sublinear improvements to accuracy. This holds true for RandAugment. For example, adding RandAugment on top of BlurPool, MixUp, and label smoothing yields no significant additional accuracy improvement.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.randaugment.RandAugment
    :noindex:

.. autofunction:: composer.algorithms.randaugment.randaugment_image
    :noindex:

.. autofunction:: composer.algorithms.randaugment.RandAugmentTransform
    :noindex:
```
