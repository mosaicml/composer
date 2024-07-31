# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper functions to perform augmentations on a :class:`PIL.Image.Image`.

Augmentations that take an intensity value are normalized on a scale of 1-10,
where 10 is the strongest and maximum value an augmentation function will accept.

Adapted from
`AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
<https://github.com/google-research/augmix/blob/master/augmentations.py>`_.

Attributes:
    AugmentationFn ((PIL.Image.Image, float) -> PIL.Image.Image):
        The type annotation for describing an augmentation function.

        Each augmentation takes a :class:`PIL.Image.Image` and an intensity level in the range ``[0, 10]``,
        and returns an augmented image.

    augmentation_sets (dict[str, list[AugmentationFn]]): The collection of all augmentations.
        This dictionary has the following entries:

        * ``augmentation_sets["safe"]`` contains augmentations that do not overlap with
            ImageNet-C/CIFAR10-C test sets.
        * ``augmentation_sets["original"]`` contains augmentations that use the original
            implementations of enhancing color, contrast, brightness, and sharpness.
        * ``augmentation_sets["all"]`` contains all augmentations.
"""
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from PIL.Image import Resampling, Transform

AugmentationFn = Callable[[Image.Image, float], Image.Image]

__all__ = [
    'AugmentationFn',
    'autocontrast',
    'equalize',
    'posterize',
    'rotate',
    'solarize',
    'shear_x',
    'shear_y',
    'translate_x',
    'translate_y',
    'color',
    'color_original',
    'contrast',
    'contrast_original',
    'brightness',
    'brightness_original',
    'sharpness',
    'sharpness_original',
    'augmentation_sets',
]


def _int_parameter(level: float, maxval: float):
    """Helper function to scale a value between ``0`` and ``maxval`` and return as an int.

    Args:
        level (float): Level of the operation that will be between ``[0, 10]``.
        maxval (float): Maximum value that the operation can have. This will be scaled to
            ``level/10``.

    Returns:
        int: The result from scaling ``maxval`` according to ``level``.
    """
    return int(level * maxval / 10)


def _float_parameter(level: float, maxval: float):
    """Helper function to scale a value between ``0`` and ``maxval`` and return as a float.

    Args:
        level (float): Level of the operation that will be between [0, 10].
        maxval (float): Maximum value that the operation can have. This will be scaled to
            ``level/10``.

    Returns:
        float: The result from scaling ``maxval`` according to ``level``.
    """
    return float(level) * maxval / 10.


def _sample_level(n: float):
    """Helper function to sample from a uniform distribution between ``0.1`` and some value ``n``."""
    return np.random.uniform(low=0.1, high=n)


def _symmetric_sample(level: float):
    """Helper function to sample from a symmetric distribution.

    The distribution over the domain [0.1, 10] with ``median == 1`` and uniform probability of ``x | 0.1 ≤ x ≤ 1``,
    and ``x | 1 ≤ x ≤ 10``.

    Used for sampling transforms that can range from intensity 0 to infinity and for which an intensity
    of 1 meaning no change.
    """
    if np.random.uniform() > 0.5:
        return np.random.uniform(1, level)
    else:
        return np.random.uniform(1 - (0.09 * level), 1)


def autocontrast(pil_img: Image.Image, level: float = 0.0):
    """Autocontrast an image.

    .. seealso:: :func:`PIL.ImageOps.autocontrast`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity.
    """
    del level  # unused
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img: Image.Image, level: float):
    """Equalize an image.

    .. seealso:: :func:`PIL.ImageOps.equalize`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity.
    """
    del level  # unused
    return ImageOps.equalize(pil_img)


def posterize(pil_img: Image.Image, level: float):
    """Posterize an image.

    .. seealso:: :func:`PIL.ImageOps.posterize`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img: Image.Image, level: float):
    """Rotate an image.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    degrees = _int_parameter(_sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Resampling.BILINEAR)


def solarize(pil_img: Image.Image, level: float):
    """Solarize an image.

    .. seealso:: :func:`PIL.ImageOps.solarize`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img: Image.Image, level: float):
    """Shear an image horizontally.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Transform.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Resampling.BILINEAR,
    )


def shear_y(pil_img: Image.Image, level: float):
    """Shear an image vertically.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Transform.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Resampling.BILINEAR,
    )


def translate_x(pil_img: Image.Image, level: float):
    """Shear an image horizontally.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Transform.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Resampling.BILINEAR,
    )


def translate_y(pil_img: Image.Image, level: float):
    """Shear an image vertically.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _int_parameter(_sample_level(level), pil_img.size[1] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        pil_img.size,
        Transform.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Resampling.BILINEAR,
    )


# The following augmentations overlap with corruptions in the ImageNet-C/CIFAR10-C test
# sets. Their original implementations also have an intensity sampling scheme that
# samples a value bounded by 0.118 at a minimum, and a maximum value of intensity*0.18+
# 0.1, which ranged from 0.28 (intensity = 1) to 1.9 (intensity 10). These augmentations
# have different effects depending on whether they are < 0 or > 0, so the original
# sampling scheme does not make sense to me. Accordingly, I replaced it with the
# _symmetric_sample() above.


def color(pil_img: Image.Image, level: float):
    """Enhance color on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Color`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    return ImageEnhance.Color(pil_img).enhance(level)


def color_original(pil_img: Image.Image, level: float):
    """Enhance color on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso :class:`PIL.ImageEnhance.Color`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img: Image.Image, level: float):
    """Enhance contrast on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Contrast`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    return ImageEnhance.Contrast(pil_img).enhance(level)


def contrast_original(pil_img: Image.Image, level: float):
    """Enhance contrast on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso:: :class:`PIL.ImageEnhance.Contrast`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img: Image.Image, level: float):
    """Enhance brightness on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Brightness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should be
            in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    # Reduce intensity of brightness increases
    if level > 1:
        level = level * .75
    return ImageEnhance.Brightness(pil_img).enhance(level)


def brightness_original(pil_img: Image.Image, level: float):
    """Enhance brightness on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso:: :class:`PIL.ImageEnhance.Brightness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img: Image.Image, level: float):
    """Enhance sharpness on an image.

    .. seealso:: :class:`PIL.ImageEnhance.Sharpness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _symmetric_sample(level)
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def sharpness_original(pil_img: Image.Image, level: float):
    """Enhance sharpness on an image, following the
    corruptions in the ImageNet-C/CIFAR10-C test sets.

    .. seealso:: :class:`PIL.ImageEnhance.Sharpness`.

    Args:
        pil_img (PIL.Image.Image): The image.
        level (float): The intensity, which should
            be in ``[0, 10]``.
    """
    level = _float_parameter(_sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentation_sets = {
    'all': [
        autocontrast,
        equalize,
        posterize,
        rotate,
        solarize,
        shear_x,
        shear_y,
        translate_x,
        translate_y,
        color,
        contrast,
        brightness,
        sharpness,
    ],
    # Augmentations that don't overlap with ImageNet-C/CIFAR10-C test sets
    'safe': [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y],
    # Augmentations that use original implementations of color, contrast, brightness, and sharpness
    'original': [
        autocontrast,
        equalize,
        posterize,
        rotate,
        solarize,
        shear_x,
        shear_y,
        translate_x,
        translate_y,
        color_original,
        contrast_original,
        brightness_original,
        sharpness_original,
    ],
}
