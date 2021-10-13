# Copyright 2021 MosaicML. All Rights Reserved.

# Modified from https://github.com/google-research/augmix/blob/master/augmentations.py

"""Augmentation primitives for use in AugMix and RandAugment. Augmentation intensities
    are normalized on a scale of 1-10, where 10 is the strongest and maximum value an
    augmentation function will accept.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def int_parameter(level, maxval):
    """Helper function to scale a value between 0 and maxval and return as an int.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale a value between 0 and maxval and return as a float.
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / 10.


def sample_level(n):
    """Helper function to sample from a uniform distribution between 0.1 and some value n."""
    return np.random.uniform(low=0.1, high=n)


def symmetric_sample(level):
    """Helper function to sample from a distribution over the domain [0.1, 10] with
        median == 1 and uniform probability of x | 0.1 ≤ x ≤ 1, and x | 1 ≤ x ≤ 10. Used
        for sampling transforms that can range from intensity 0 to infinity, and for
        which an intensity of 1 == no change.
    """
    if np.random.uniform() > 0.5:
        return np.random.uniform(1, level)
    else:
        return np.random.uniform(1 - (0.09 * level), 1)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


# The following augmentations overlap with corruptions in the ImageNet-C/CIFAR10-C test
# sets. Their original implementations also have an intensity sampling scheme that
# samples a value bounded by 0.118 at a minimum, and a maximum value of intensity*0.18+
# 0.1, which ranged from 0.28 (intensity = 1) to 1.9 (intensity 10). These augmentations
# have different effects depending on whether they are < 0 or > 0, so the original
# sampling scheme does not make sense to me. Accordingly, I replaced it with the
# symmetric_sample() above.


def color(pil_img, level):
    level = symmetric_sample(level)
    return ImageEnhance.Color(pil_img).enhance(level)


def color_original(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img, level):
    level = symmetric_sample(level)
    return ImageEnhance.Contrast(pil_img).enhance(level)


def contrast_original(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img, level):
    level = symmetric_sample(level)
    # Reduce intensity of brightness increases
    if level > 1:
        level = level * .75
    return ImageEnhance.Brightness(pil_img).enhance(level)


def brightness_original(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img, level):
    level = symmetric_sample(level)
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def sharpness_original(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentation_sets = {
    "all": [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color,
        contrast, brightness, sharpness
    ],
    # Augmentations that don't overlap with ImageNet-C/CIFAR10-C test sets
    "safe": [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y],
    # Augmentations that use original implementations of color, contrast, brightness, and sharpness
    "original": [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color_original,
        contrast_original, brightness_original, sharpness_original
    ],
}
