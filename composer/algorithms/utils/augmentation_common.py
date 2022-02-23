# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Callable, Iterable, Type, TypeVar, cast

import torch
from PIL.Image import Image as PillowImage
from torchvision import transforms

_InputImgT = TypeVar("_InputImgT", torch.Tensor, PillowImage)
_OutputImgT = TypeVar("_OutputImgT", torch.Tensor, PillowImage)


def image_as_type(image: _InputImgT, typ: Type[_OutputImgT]) -> _OutputImgT:
    """Converts between :class:`torch.Tensor` and
            :class:`PIL.Image.Image` image representations

    Args:
        image: a single image represented as a :class:`torch.Tensor` or
            :class:`PIL.Image.Image`
        typ: type of the copied image. Must be :class:`torch.Tensor` or
            :class:`PIL.Image.Image`

    Returns:
        A copy of ``image`` with type ``typ``

    Raises:
        TypeError: if ``typ`` is not one of :class:`torch.Tensor` or
            :class:`PIL.Image.Image`.
    """
    if isinstance(image, typ):
        return image
    if not issubclass(typ, (torch.Tensor, PillowImage)):
        raise TypeError(f"Only typ={{torch.Tensor, Image}} is supported; got {typ}")

    if isinstance(image, PillowImage):
        return transforms.functional.to_tensor(image)  # PIL -> Tensor
    # if we got to here, image is tensor, and requested type is PIL
    return transforms.functional.to_pil_image(image)  # Tensor -> PIL


def image_typed_and_shaped_like(image: _InputImgT,
                                reference_image: _OutputImgT) -> _OutputImgT:  # type: ignore[reportUnusedFunction]
    """Creates a copy of an image-like object with the same type and shape as another.

    Args:
        image: A tensor or PIL image
        reference_image: Another tensor or PIL image with the same shape

    Returns:
        A copy of ``image`` with the same type and shape as ``reference_image``.

    Raises:
        See :func:`image_as_type`.
    """
    typ = type(reference_image)
    ret_image = cast(_OutputImgT, image_as_type(image, typ=typ))
    if issubclass(typ, torch.Tensor):
        new_shape = cast(torch.Tensor, reference_image).shape
        ret_image = cast(torch.Tensor, ret_image).reshape(new_shape)
    return cast(_OutputImgT, ret_image)


def _map_pillow_function(f_pil: Callable[[PillowImage], PillowImage], imgs: _OutputImgT) -> _OutputImgT:
    """Lifts a function that requires pillow images to also work on tensors"""
    single_image_input = not isinstance(imgs, Iterable)
    single_image_input |= isinstance(imgs, torch.Tensor) and imgs.ndim == 3
    if single_image_input:
        imgs = [imgs]

    imgs_pil = [image_as_type(img, PillowImage) for img in imgs]
    imgs_out_pil = [f_pil(img_pil) for img_pil in imgs_pil]
    imgs_out = [image_as_type(img_pil, type(imgs[0])) for img_pil in imgs_out_pil]

    if isinstance(imgs, torch.Tensor) and imgs.ndim == 4:  # batch of imgs
        imgs_out = [torch.unsqueeze(img, 0) for img in imgs_out]
        imgs_out = torch.cat(imgs_out, dim=0)
    if single_image_input:
        imgs_out = imgs_out[0]
    return imgs_out
