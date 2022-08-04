# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Core CopyPaste classes and functions."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple, Union

from numpy.random import default_rng
import numpy as np
import random
import torch
import torchvision.transforms as T

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

log = logging.getLogger(__name__)

__all__ = ['CopyPaste', 'copypaste_batch']

_MAX_TORCH_SEED = 0xffff_ffff_ffff_ffff


def copypaste_batch(images, masks, configs):
    """
    copy-paste is an augmentation method. Two images are randomly chosen (with 
    replacement) from a given batch of images and their corresponding masks, i.e.,
    the source and the target target. A number of instances from the source are
    selected to be copied into the target. The number of copied instanes is always
    less than a number determined by the minimum of ``max_copied_instance`` and 
    total number of instances in the source. A set of instances are then randomly
    chosen (without replacement) from the source. Each instance goes through a set
    of transformation and jittering, e.g., horizontal flipping, rescaling, and 
    cropping. The resulting "jittered" instance is then checked for its area and 
    transformed instances with an area smaller than the configured threshold. 
    The resulting jittered instances are then pasted on a random position on 
    the target. Same procedure is also applied on the masks of the source and 
    the target. If a copied instances mask collides with an exisiting mask on 
    the target, the copied instance's mask overlays the original mask on the 
    target.

    Args:
        input (torch.Tensor): input tensor of shape ``(N, C, H, W)``.
        target (torch.Tensor): target tensor of shape ``(N, H, W)``.   
        configs (dict): dictionary containing the configurable hyperparameters.    

    Returns:
        out_images (torch.Tensor): batch of images after copypaste has been
            applied.
        out_masks (torch.Tensor): batch of corresponding masks after applying 
        copypaste to them

    Example:
        .. testcode::

            import torch
            from composer.functional import copypaste_batch

            N, C, H, W = 2, 3, 4, 5
            num_classes = 10
            configs = {
                "p": 1.0,
                "max_copied_instances": None,
                "area_threshold": 100,
                "padding_factor": 0.5,
                "jitter_scale": (0.01, 0.99),
                "jitter_ratio": (1.0, 1.0),
                "p_flip": 1.0,
                "bg_color": 0
            }
            X = torch.randn(N, C, H, W)
            y = torch.randint(num_classes, size=(N, H, W))
            out_images, out_masks = cutmix_batch(X, y, configs)
    """
    batch_idx = 0
    out_images = torch.zeros_like(images)
    out_masks = torch.zeros_like(masks)

    assert images.size(dim=0) == masks.size(
        dim=0), "Number of images and masks in the batch do not match!"
    batch_size = images.size(dim=0)

    while batch_idx < batch_size:
        [i, j] = np.random.randint(0, high=batch_size, size=2)

        num_instances = _count_instances(masks[i])
        num_copied_instances = random.randint(0, num_instances)
        if configs["max_copied_instances"] is not None:
            num_copied_instances = min(
                num_copied_instances, configs["max_copied_instances"])

        src_instance_ids = _get_instance_ids(
            masks[i], num_copied_instances, configs["bg_color"])

        trg_image = images[j]
        trg_mask = masks[j]

        if random.uniform(0, 1) < configs["p"]:
            for src_instance_id in src_instance_ids:
                trg_image, trg_mask = _copypaste_instance(
                    images[i], masks[i], trg_image, trg_mask, src_instance_id, configs)

        out_images[batch_idx] = trg_image
        out_masks[batch_idx] = trg_mask
        batch_idx += 1

    return out_images, out_masks


class CopyPaste(Algorithm):
    """
    Randomly pastes objects onto an image.

    Args:
        p (float, optional): Probability of applyig copy-paste augmentation on a
            pair of randomly chosen source and target samples. Default: ``0.5``
        max_copied_instances (int | None, optional): Maximum number of instances
            to be copied from a randomly chosen source sample into another 
            randomly chosen target sample. If this value is greater than the total 
            number of instances in the source sample, it is overridden by the 
            total number of instances in the source sample. If it is set to 
            ``None``, the total number of instances in the source sample is set to 
            be the limit. Default: ``None``.
        area_threshold (int, optional): Minimum area (in pixels) of an augmented
            instance to be considered a valid instance. Augmented instances with 
            an area smaller than this threshold are removed from the sample. 
            Default:``25``.
        padding_factor (float, optional): The source sample is padded by this 
            ratio before applying large scale jittering to it. Default: ``0.5``.
        jitter_scale (Tuple[float, float], optional): Determines the scale used 
            in the large scale jittering of the source instance. Specifies the 
            lower and upper bounds for the random area of the crop, before 
            resizing. The scale is defined with respect to the area of the 
            original image. Default: ``(0.01, 0.99)``.
        jitter_ratio (Tuple[float, float], optional): Determines the ratio used in 
            the large scale jittering of the source instance. Lower and upper 
            bounds for the random aspect ratio of the crop, before resizing. 
            Default: ``(1.0, 1.0)``.
        p_flip (float, optional): Probability of applying horizontal flipping 
            during large scale jittering of the source instance. Default: ``0.9``.
        bg_color (int, optional): Class label (pixel value) of the background 
            class. Default: ``-1``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key 
            that indexes to the input from the batch. Can also be a pair of get 
            and set functions, where the getter is assumed to be first in the 
            pair.  The default is 0, which corresponds to any sequence, where the 
            first element is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key 
            that indexes to the target from the batch. Can also be a pair of get 
            and set functions, where the getter is assumed to be first in the 
            pair. The default is 1, which corresponds to any sequence, where the 
            second element is the target. Default: ``1``.

    Example:
        .. testcode::

            from composer.algorithms import CopyPaste
            algorithm = CopyPaste()
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[algorithm],
                optimizers=[optimizer]
            )

    """

    def __init__(
        self,
        p=0.5,
        max_copied_instances=None,
        area_threshold=25,
        padding_factor=0.5,
        jitter_scale=(0.01, 0.99),
        jitter_ratio=(1.0, 1.0),
        p_flip=0.9,
        bg_color=-1,
        input_key: Union[str, int, Tuple[Callable, Callable], Any] = 0,
        target_key: Union[str, int, Tuple[Callable, Callable], Any] = 1,
    ):
        self.input_key = input_key
        self.target_key = target_key
        self.configs = {
            "p": p,
            "max_copied_instances": max_copied_instances,
            "area_threshold": area_threshold,
            "padding_factor": padding_factor,
            "jitter_scale": jitter_scale,
            "jitter_ratio": jitter_ratio,
            "p_flip": p_flip,
            "bg_color": bg_color
        }

    def match(self, event: Event, state: State) -> bool:
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        images = state.batch_get_item(key=self.input_key)
        masks = state.batch_get_item(key=self.target_key)

        out_images, out_masks = copypaste_batch(images, masks, self.configs)

        state.batch_set_item(key=self.input_key, value=out_images)
        state.batch_set_item(key=self.target_key, value=out_masks)


def _copypaste_instance(src_image, src_mask, trg_image, trg_mask, src_instance_id, configs):
    """Applies copy-paste augmentation on a set of source and target samples. The 
    instance identified by ``src_instance_id`` is selected from the source sample 
    to be copied to the target sample.

    Args:
        src_image (torch.Tensor): Source image of shape ``(C, H, W)``,
            C is the number of channels.
        src_mask (torch.Tensor): Source mask of shape ``(H, W)``,
        trg_image (torch.Tensor): Target image of shape ``(C, H, W)``,
            C is the number of channels.
        trg_mask (torch.Tensor): Target mask of shape ``(H, W)``,
        src_instance_id (int): Class ID of the randmoly chosen instance to be 
            copied from the source sample into the target sample.
        configs (dict): Configurable hyperparameters.

    Returns:
        trg_image (torch.Tensor): Augmented target image of shape ``(C, H, W)``,
            C is the number of channels.
        trg_mask (torch.Tensor): Augmented target mask of shape ``(H, W)``,
    """
    zero_tensor = torch.zeros(
        1, dtype=src_image.dtype, device=src_image.device)
    bg_color = configs["bg_color"]

    src_instance_mask = _parse_mask_by_id(src_mask, src_instance_id, bg_color)
    src_instance = torch.where(
        src_instance_mask == bg_color, zero_tensor, src_image)

    [src_instance, src_instance_mask] = _jitter_instance(
        [src_instance, torch.unsqueeze(src_instance_mask, dim=0)], configs)
    src_instance_mask = torch.squeeze(src_instance_mask)

    trg_image = torch.where(src_instance_mask ==
                            bg_color, trg_image, zero_tensor)
    trg_image += src_instance
    trg_mask = torch.where(src_instance_mask == bg_color,
                           trg_mask, src_instance_mask)

    return trg_image, trg_mask


def _get_jitter_transformations(is_mask, padding_size, crop_size, configs):
    """A wrapper around ``torchvision.transforms.transforms``. Generates a Torch transformation object to be used in large scale jittering.

    Args:
        is_mask (int): Identifier that indicates if the transformation is used on 
            a mask or an image.
        padding_size (int or sequence): Padding on each border. If a single int is 
            provided this is used to pad all borders. If sequence of length 2 is provided this is the padding on left/right and top/bottom respectively.
        size (int or sequence): Expected output size of the crop, for each edge. If 
            size is an int instead of sequence like (h, w), a square output size 
            ``(size, size)`` is made. If provided a sequence of length 1, it will 
            be interpreted as (size[0], size[0]).
        configs (dict): Configurable hyperparameters.


    Returns:
        trns (torchvision.transforms.transforms): An object of trochvision 
            transformations.
    """
    fill = 0
    if is_mask:
        fill = configs["bg_color"]

    trns = T.Compose([
        T.Pad(padding=padding_size, fill=fill, padding_mode="constant"),
        T.RandomResizedCrop(size=crop_size, scale=configs["jitter_scale"],
                            ratio=configs["jitter_ratio"], interpolation=T.InterpolationMode.NEAREST),
        T.RandomHorizontalFlip(p=configs["p_flip"])
    ])

    return trns


def _jitter_instance(arrs, configs):
    """Applies transformations on a tuple of image and mask. 

    Args:
        arrs (sequence): Sequence containing the image and mask tensors. Element 0
            always contains the image and element 1 contains the mask.
        configs (dict): Configurable hyperparameters.


    Returns:
        out (sequence): Sequence containing the jittered (transformed) image and mask 
            tensors. Element 0 always contains the image and element 1 contains the 
            mask.
    """

    out = []
    jitter_seed = random.randint(0, _MAX_TORCH_SEED)

    padding_size = (int(configs["padding_factor"] * arrs[0].size(dim=1)),
                    int(configs["padding_factor"] * arrs[0].size(dim=2)))
    crop_size = (arrs[0].size(dim=1), arrs[0].size(dim=2))

    for idx, arr in enumerate(arrs):
        torch.random.manual_seed(jitter_seed)
        random.seed(jitter_seed)
        trns = _get_jitter_transformations(
            idx, padding_size, crop_size, configs)
        transformed = trns(arr)

        out.append(transformed)

    if _ignore_instance(out[-1], configs):
        return arrs

    return out


def _ignore_instance(mask, configs):
    """Compares the area of the mask with a threshold determined by the parameters 
    in ``configs``. 

    Args:
        mask (torch.Tensore): Tensor of the instance mask.
        configs (dict): Configurable hyperparameters.


    Returns:
        ignore (bool): A boolean flag determining if the mask must be ignored or not.
    """
    uniques = torch.unique(mask, return_counts=True)

    mask_area = 0
    for i, count in enumerate(uniques[1]):
        if uniques[0][i] != configs["bg_color"]:
            mask_area += count

    return bool(int(mask_area) < configs["area_threshold"])


def _count_instances(input_tensor):
    """Counts the total number of non-background class IDs in a mask tensor. 

    Args:
        input_tensor (torch.Tensore): Tensor of mask with shape ``(H, W)``.


    Returns:
        count (int): Number of non-background class IDs.
    """
    unique_class_ids = torch.unique(input_tensor)

    return (len(unique_class_ids) - 1)


def _get_instance_ids(input_tensor, num_instances, filtered_id):
    """Generates a list of randomly selected (without replacement) 
    non-background class IDs in a mask tensor. 

    Args:
        input_tensor (torch.Tensore): Tensor of mask with shape ``(H, W)``.
        num_instances (int): Number of instances to be randomly selected from 
            ``input_tensor``.
        filtered_id (int): Class ID of the background class


    Returns:
        indices (sequence): a list of randomly selected (without replacement) 
            non-background class IDs in a mask tensor.
    """

    instance_ids = torch.unique(input_tensor)

    npy = instance_ids.cpu().numpy()
    npy = np.delete(npy, np.where(npy == filtered_id))

    rng = default_rng()
    indices = rng.choice(len(npy), size=num_instances, replace=False)

    return npy[indices]


def _parse_mask_by_id(mask, idx, background_color=-1):
    """Extracts an instance indicated by ``idx`` from an input mask tensor. 

    Args:
        mask (torch.Tensore): Tensor of mask with shape ``(H, W)``.
        idx (int): Class ID of the desired instance to be extracted from ``mask``
        background_color (int): Class ID of the background class.


    Returns:
        parsed_mask (torch.Tensore): Tensor of mask with shape ``(H, W)`` that only 
            contains the instance indicated by ``idx``.
    """
    parsed_mask = torch.where(mask == idx, idx, background_color)

    return parsed_mask
