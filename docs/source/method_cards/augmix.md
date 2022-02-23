# AugMix

![aug_mix.png](https://storage.googleapis.com/docs.mosaicml.com/images/methods/aug_mix.png)

Fig. 4 from [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://arxiv.org/abs/1912.02781), by Hendrycks et al. (2020).

Tags: `Vision`, `Increased Accuracy`, `Increased CPU Usage`, `Method`, `Augmentation`, `Regularization`

## TL;DR

For each data sample, AugMix creates an *augmentation chain* by sampling `depth` image augmentations from a set (e.g. translation, shear, contrast) and applies them sequentially with randomly sampled intensity. This is repeated `width` times in parallel to create `width` different augmentation chains. The augmented images are then combined via a random convex combination to yield a single augmented image, which is in turn combined via a random convex combination sampled from a Beta(`alpha`, `alpha`) distribution with the original image.

## Attribution

[AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](http://arxiv.org/abs/1912.02781) by Dan Hendrycks, Norman Mu, Ekin D. Cubuk, Barret Zoph, Justin Gilmer, and Balaji Lakshminarayanan. Published at ICLR 2020. 

## Applicable Settings

AugMix produces image augmentations, and hence is only applicable to vision tasks. Hendrycks et al. show in their paper introducing the method that it improves accuracy and robustness in image classification tasks.

## Hyperparameters

- `severity` - The maximum possible intensity of each augmentation.
- `depth` - The number of augmentations to perform in each augmentation chain.
- `width` - The number of parallel augmentation chains. A value of -1 means width will be randomly sampled from the uniform distribution {1, 2, 3} for each data sample. *k* in Hendrycks et al., 2020.
- `alpha` - Mixing parameter for clean vs. augmented images.
- `augmentation_set` - The set of augmentations to sample from. `all` is the set  {`translate_x`, `translate_y`, `shear_x`, `shear_y`, `rotate`, `solarize`, `posterize`, `equalize`, `autocontrast`, `color`, `contrast`, `brightness`, `sharpness`}. `safe` excludes `color`, `contrast`, `brightness`, and `sharpness`, which are used as part of the CIFAR10C and ImageNetC benchmark datasets for naturalistic robustness ([Hendrycks et al., 2019](https://arxiv.org/abs/1903.12261)). `original` uses implementations of `color`, `contrast`, `brightness`, and `sharpness` that replicate [existing](https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/auto_augment.py) [implementations](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py). [See the code](https://github.com/mosaicml/mosaicml/blob/f8d2a67bb3e08b24c299dda0bf76ef64bc25db35/composer/utils/augmentation_primitives.py#L105) for more details.

## Example Effects

Hendrycks et al. report a 13.8 percentage point (pp) accuracy improvement on CIFAR10C ([a benchmark for corruption robustness](https://arxiv.org/abs/1903.12261)) over baseline for a 40-2 [Wide ResNet](https://arxiv.org/abs/1605.07146), and a 1.5-10pp improvement over other augmentation schemes. Hendrycks et al. also report a 1.5pp improvement over a baseline ResNet-50 on ImageNet, but this result utilizes AugMix in combination with a custom loss function (See **Implementation Details**). When omitting the custom loss function and using the AugMix augmentation scheme alone, we observe an accuracy gain of ~0.5pp over a baseline ResNet-50 on ImageNet. However, the increased CPU load imposed by AugMix substantially reduces throughput: using AugMix with the hyperparameters recommended by Hendrycks et al. can increase training time by 1.1-10x, depending on the hardware and model.

## Implementation Details

AugMix randomly samples image augmentations (with replacement) from the set of {translate_x, translate_y, shear_x, shear_y, rotate, solarize, equalize, posterize, autocontrast, color, brightness, contrast, sharpness}, with the intensity of each augmentation sampled uniformly from 0.1-`severity` (`severity` ≤ 10). The augmentations use the [PILLOW Image library](https://pillow.readthedocs.io/en/stable/reference/Image.html) (specifically [Pillow-SIMD](https://github.com/uploadcare/pillow-simd)), as we found [OpenCV](https://opencv.org/)-based augmentations resulted in similar or worse performance. AugMix is applied after "standard" transformations such as resizing and cropping, and before normalization. Hendrycks et al.'s original implementation of AugMix also includes a custom loss function computed across three samples (an image and two AugMix'd versions of that image). We omit this custom loss function from our AugMix implementation because it effectively triples the number of samples required for a parameter update step, imposing a significant computational burden. Our implementation, which consists only of the augmentation component of AugMix, is referred to by Hendrycks et al. as "AugmentAndMix".

## Suggested Hyperparameters

As per Hendrycks et al., we found that `width`= 3, `severity`= 3, `depth`= -1, (`depth`= -1 means that `depth` will be randomly sampled from the uniform distribution {1, 2, 3} for each data sample). `alpha`= 1 worked well for different models of the ResNet family, however we used `augmentation_set`= `all`, which is incompatible with downstream evaluation on CIFAR-10C/ImageNetC because `all` contains augmentations that are used to generate CIFAR10C/ImageNetC. Increasing `width` or `depth` significantly decreases throughput.

## Considerations

As mentioned in the "Example Effects" section, we found that AugMix can significantly decrease throughput. This is due to the increased CPU load from performing the image augmentations. We found that training time could increase by up to 10x on some hardware. The considerations for AugMix are very similar to those for RandAugment in terms of both hardware and the effect on the model: both methods use the CPU to apply image augmentations to data samples in order to regularize the model. As such, AugMix will be more useful in overparameterized regimes (i.e. larger models, smaller datasets, and longer training times) and systems with comparatively greater CPU:GPU resources. See RandAugment/Considerations for more detail.

## Composability

As general rule, combinging regularization-based methods yields sublinear improvements to accuracy. See RandAugment/Composability for more details.

---

## Code

```{eval-rst}
.. autoclass:: composer.algorithms.augmix.AugMix
    :noindex:

.. autofunction:: composer.algorithms.augmix.augmix_image
    :noindex:

.. autoclass:: composer.algorithms.augmix.AugmentAndMixTransform
    :noindex:
```