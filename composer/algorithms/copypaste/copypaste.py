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
        p (float, optional): p. Default: ``0.5``        
        max_copied_instances (int | None, optional): max_copied_instances. Default: ``None``.
        area_threshold (int, optional): area_threshold in pixels. Default: ``25``.
        padding_factor (float, optional): padding_factor. Default: ``0.5``.
        jitter_scale (Tuple[float, float], optional): jitter_scale. Default: ``(0.01, 0.99)``.
        jitter_ratio (Tuple[float, float], optional): jitter_ratio. Default: ``(1.0, 1.0)``.
        p_flip (float, optional): p_flip. Default: ``0.9``.
        bg_color (int, optional): bg_color. Default: ``-1``.
        input_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the input
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair.  The default is 0, which corresponds to any sequence, where the first element
            is the input. Default: ``0``.
        target_key (str | int | Tuple[Callable, Callable] | Any, optional): A key that indexes to the target
            from the batch. Can also be a pair of get and set functions, where the getter
            is assumed to be first in the pair. The default is 1, which corresponds to any sequence, where the second element
            is the target. Default: ``1``.

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
    uniques = torch.unique(mask, return_counts=True)

    mask_area = 0
    for i, count in enumerate(uniques[1]):
        if uniques[0][i] != configs["bg_color"]:
            mask_area += count

    return bool(int(mask_area) < configs["area_threshold"])


def _count_instances(input_tensor):
    unique_class_ids = torch.unique(input_tensor)

    return (len(unique_class_ids) - 1)


def _get_instance_ids(input_tensor, num_instances, filtered_id):
    instance_ids = torch.unique(input_tensor)

    npy = instance_ids.cpu().numpy()
    npy = np.delete(npy, np.where(npy == filtered_id))

    rng = default_rng()
    indices = rng.choice(len(npy), size=num_instances, replace=False)

    return npy[indices]


def _parse_mask_by_id(mask, idx, background_color=-1):
    parsed_mask = torch.where(mask == idx, idx, background_color)

    return parsed_mask
