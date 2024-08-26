# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Iterable, TypeVar, cast

import torch
import torchvision.transforms.functional
from PIL.Image import Image as PillowImage

_InputImgT = TypeVar('_InputImgT', torch.Tensor, PillowImage)
_OutputImgT = TypeVar('_OutputImgT', torch.Tensor, PillowImage)


def image_as_type(image: _InputImgT, typ: type[_OutputImgT]) -> _OutputImgT:
    """Converts between :class:`torch.Tensor` and :class:`PIL.Image.Image` image representations.

    Args:
        image (torch.Tensor | PIL.Image.Image): A single image
            represented as a :class:`PIL.Image.Image` or
            a rank 2 or rank 3 :class:`torch.Tensor` in ``HW`` or ``CHW`` format.
            A rank 4 or higher tensor can also be provided as long as no type
            conversion is needed; in this case, the input tensor will be
            returned. This case is allowed so that functions that natively
            operate on batch tensors can safely call
            ``image_as_type(image, torch.Tensor)`` without additional error
            and type checking.
        typ (torch.Tensor | PIL.Image.Image): Type of the
            copied image. Must be :class:`PIL.Image.Image` or :class:`torch.Tensor`.

    Returns:
        A copy of ``image`` with type ``typ``.

    Raises:
        TypeError: if ``typ`` is not one of :class:`torch.Tensor` or
            :class:`PIL.Image.Image`.
        ValueError: if ``image`` cannot be converted to the ``typ``,
            such as when requesting conversion of a rank 4 tensor to
            :class:`PIL.Image.Image`.

    """
    if isinstance(image, typ):
        return image
    if not typ in (torch.Tensor, PillowImage):
        raise TypeError(f'Only typ={{torch.Tensor, Image}} is supported; got {typ}')

    if typ is torch.Tensor:
        return cast(_OutputImgT, torchvision.transforms.functional.to_tensor(image))  # type: ignore PIL -> Tensor
    return cast(_OutputImgT, torchvision.transforms.functional.to_pil_image(image))  # Tensor -> PIL


def map_pillow_function(f_pil: Callable[[PillowImage], PillowImage], imgs: _OutputImgT) -> _OutputImgT:
    """Lifts a function that requires pillow images to also work on tensors.

    Args:
        f_pil ((PIL.Image.Image) -> PIL.Image.Image): A callable that takes maps :class:`PIL.Image.Image` objects.
            to other :class:`PIL.Image.Image` objects.
        imgs (torch.Tensor | PIL.Image.Image): a :class:`PIL.Image.Image` or a :class:`torch.Tensor` in ``HW``,
            ``CHW`` or ``NCHW`` format.

    Returns:
        The result of applying ``f_pil`` to each image in ``imgs``, converted
        back to the same type and (if applicable) tensor layout as ``imgs``.
    """
    single_image_input = not isinstance(imgs, Iterable)
    single_image_input |= isinstance(imgs, torch.Tensor) and imgs.ndim == 3
    imgs_as_iterable = [imgs] if single_image_input else imgs
    imgs_as_iterable = cast(type(imgs_as_iterable), imgs_as_iterable)

    imgs_pil = [image_as_type(img, PillowImage) for img in imgs_as_iterable]
    imgs_out_pil = [f_pil(img_pil) for img_pil in imgs_pil]
    imgs_out = [image_as_type(img_pil, type(imgs_as_iterable[0])) for img_pil in imgs_out_pil]

    if isinstance(imgs, torch.Tensor) and imgs.ndim == 4:  # batch of imgs
        imgs_out = [torch.unsqueeze(cast(torch.Tensor, img), 0) for img in imgs_out]
        imgs_out = torch.cat(imgs_out, dim=0)
    if single_image_input:
        imgs_out = imgs_out[0]
    imgs_out = cast(_OutputImgT, imgs_out)
    return imgs_out
