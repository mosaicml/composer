from typing import Type, TypeVar, cast

import torch
from PIL.Image import Image as PillowImage
from torchvision import transforms

_InputImgT = TypeVar("_InputImgT", torch.Tensor, PillowImage)
_OutputImgT = TypeVar("_OutputImgT", torch.Tensor, PillowImage)


def image_as_type(image: _InputImgT,
                  need_type: Type[_OutputImgT]) -> _OutputImgT:
    """Creates a copy of an image-like object with a different type

    Args:
        image: a tensor or PIL image. Batches of images in NCHW format are
            also supported.
        need_type: type of the copied image

    Returns:
        A copy of ``image`` with type ``need_type``.

    Raises:
        RuntimeError: if ``image`` is batch of images, but ``need_type`` is
            a PIL image. We could convert to a batch of PIL images, but we
            instead fail fast since this is not expected behavior at any
            call sites (within data augmenation functions).
        TypeError: if ``need_type`` is not one of :class:`torch.Tensor` or
            :class:`PIL.Image.Image`.
    """
    if isinstance(image, need_type):
        return image

    # fail fast if attempting to convert a batch of images to a single PIL image
    is_batch = isinstance(image, torch.Tensor) and (image.ndim == 4) and (image.shape[0] > 1)
    if is_batch and issubclass(need_type, PillowImage):
        raise RuntimeError(f"Could not convert batch of images to PIL image")

    if not issubclass(need_type, (torch.Tensor, PillowImage)):
        raise TypeError(f"Only need_type={{torch.Tensor, Image}} is supported; got {need_type}")

    if isinstance(image, PillowImage):
        return transforms.functional.to_tensor(image)  # PIL -> Tensor
    return transforms.functional.to_pil_image(image)   # Tensor -> PIL


def image_typed_and_shaped_like(image: _InputImgT, reference_image: _OutputImgT) -> _OutputImgT:  # type: ignore[reportUnusedFunction]
    """Creates a copy of an image-like object with the same type and shape as another.

    Args:
        image: A tensor or PIL image
        reference_image: Another tensor or PIL image with the same shape

    Returns:
        A copy of ``image`` with the same type and shape as ``reference_image``.

    Raises:
        See :func:`image_as_type`.
    """
    need_type = type(reference_image)
    ret_image = cast(_OutputImgT, image_as_type(image, need_type=need_type))
    if issubclass(need_type, torch.Tensor):
        new_shape = cast(torch.Tensor, reference_image).shape
        ret_image = cast(torch.Tensor, ret_image).reshape(new_shape)
    return cast(_OutputImgT, ret_image)
