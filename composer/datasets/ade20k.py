# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""ADE20K Semantic segmentation and scene parsing dataset.

Please refer to the `ADE20K dataset <https://groups.csail.mit.edu/vision/datasets/ADE20K/>`_ for more details about this
dataset.
"""

import os
from math import ceil
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from composer.core import DataSpec, MemoryFormat
from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.utils import MissingConditionalImportError, dist

__all__ = [
    'ADE20k', 'build_ade20k_dataloader', 'build_streaming_ade20k_dataloader', 'build_synthetic_ade20k_dataloader'
]

IMAGENET_CHANNEL_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)
IMAGENET_CHANNEL_STD = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def build_ade20k_transformations(split,
                                 base_size: int = 512,
                                 min_resize_scale: float = 0.5,
                                 max_resize_scale: float = 2.0,
                                 final_size: int = 512):
    """Builds the transformations for the ADE20k dataset.

       Args:
           base_size (int): Initial size of the image and target before other augmentations. Default: ``512``.
           min_resize_scale (float): The minimum value the samples can be rescaled. Default: ``0.5``.
           max_resize_scale (float): The maximum value the samples can be rescaled. Default: ``2.0``.
           final_size (int): The final size of the image and target. Default: ``512``.

       Returns:
           both_transforms (torch.nn.Module): Transformations to apply to a 2-tuple containing the input image and the
               target semantic segmentation mask.
           image_transforms (torch.nn.Module): Transformations to apply to the input image only.
           target_transforms (torch.nn.Module): Transformations to apply to the target semantic segmentation mask only.
    """
    if split == 'train':
        both_transforms = torch.nn.Sequential(
            RandomResizePair(
                min_scale=min_resize_scale,
                max_scale=max_resize_scale,
                base_size=(base_size, base_size),
            ),
            RandomCropPair(
                crop_size=(final_size, final_size),
                class_max_percent=0.75,
                num_retry=10,
            ),
            RandomHFlipPair(),
        )

        # Photometric distoration values come from mmsegmentation:
        # https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861
        r_mean, g_mean, b_mean = IMAGENET_CHANNEL_MEAN
        image_transforms = torch.nn.Sequential(
            PhotometricDistoration(brightness=32. / 255, contrast=0.5, saturation=0.5, hue=18. / 255),
            PadToSize(size=(final_size, final_size), fill=(int(r_mean), int(g_mean), int(b_mean))))

        target_transforms = PadToSize(size=(final_size, final_size), fill=0)
    else:
        both_transforms = None
        image_transforms = transforms.Resize(size=(final_size, final_size), interpolation=TF.InterpolationMode.BILINEAR)
        target_transforms = transforms.Resize(size=(final_size, final_size), interpolation=TF.InterpolationMode.NEAREST)
    return both_transforms, image_transforms, target_transforms


def build_ade20k_dataloader(
    global_batch_size: int,
    datadir: str,
    *,
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    base_size: int = 512,
    min_resize_scale: float = 0.5,
    max_resize_scale: float = 2.0,
    final_size: int = 512,
    ignore_background: bool = True,
    **dataloader_kwargs,
):
    """Builds an ADE20k dataloader.

    Args:
        global_batch_size (int): Global batch size.
        datadir (str): Path to location of dataset.
        split (str): The dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        drop_last (bool): Whether to drop last samples. Default: ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
        base_size (int): Initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): The minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): The maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): The final size of the image and target. Default: ``512``.
        ignore_background (bool): If true, ignore the background class when calculating the training loss.
            Default: ``true``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    both_transforms, image_transforms, target_transforms = build_ade20k_transformations(
        split=split,
        base_size=base_size,
        min_resize_scale=min_resize_scale,
        max_resize_scale=max_resize_scale,
        final_size=final_size)

    dataset = ADE20k(datadir=datadir,
                     split=split,
                     both_transforms=both_transforms,
                     image_transforms=image_transforms,
                     target_transforms=target_transforms)

    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)
    device_transform_fn = NormalizationFn(mean=IMAGENET_CHANNEL_MEAN,
                                          std=IMAGENET_CHANNEL_STD,
                                          ignore_background=ignore_background)

    return DataSpec(
        dataloader=DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              drop_last=drop_last,
                              collate_fn=pil_image_collate,
                              **dataloader_kwargs),
        device_transforms=device_transform_fn,
    )


def build_streaming_ade20k_dataloader(
    global_batch_size: int,
    remote: str,
    *,
    local: str = '/tmp/mds-cache/mds-ade20k/',
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    base_size: int = 512,
    min_resize_scale: float = 0.5,
    max_resize_scale: float = 2.0,
    final_size: int = 512,
    ignore_background: bool = True,
    predownload: Optional[int] = 100_000,
    keep_zip: Optional[bool] = None,
    download_retry: int = 2,
    download_timeout: float = 60,
    validate_hash: Optional[str] = None,
    shuffle_seed: Optional[int] = None,
    num_canonical_nodes: Optional[int] = None,
    **dataloader_kwargs: Dict[str, Any],
):
    """Build an ADE20k streaming dataset.

    Args:
        global_batch_size (int): Global batch size.
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
            Default: ``'/tmp/mds-cache/mds-ade20k/```.
        split (str): The dataset split to use, either 'train' or 'val'. Default: ``'train```.
        base_size (int): Initial size of the image and target before other augmentations. Default: ``512``.
        min_resize_scale (float): The minimum value the samples can be rescaled. Default: ``0.5``.
        max_resize_scale (float): The maximum value the samples can be rescaled. Default: ``2.0``.
        final_size (int): The final size of the image and target. Default: ``512``.
        ignore_background (bool): If true, ignore the background class when calculating the training loss.
            Default: ``true``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep iff remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int, optional): Seed for shuffling, or ``None`` for random seed. Defaults to
            ``None``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            Defaults to ``None``, which is interpreted as the number of nodes of the initial run.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()

    try:
        from streaming.vision import StreamingADE20K
    except ImportError as e:
        raise MissingConditionalImportError(extra_deps_group='streaming', conda_package='mosaicml-streaming') from e

    # Build the sets of transformations for ADE20k
    joint_transform, image_transform, target_transform = build_ade20k_transformations(
        split=split,
        base_size=base_size,
        min_resize_scale=min_resize_scale,
        max_resize_scale=max_resize_scale,
        final_size=final_size,
    )

    dataset = StreamingADE20K(
        local=local,
        remote=remote,
        split=split,
        shuffle=shuffle,
        joint_transform=joint_transform,
        transform=image_transform,
        target_transform=target_transform,
        predownload=predownload,
        keep_zip=keep_zip,
        download_retry=download_retry,
        download_timeout=download_timeout,
        validate_hash=validate_hash,
        shuffle_seed=shuffle_seed,
        num_canonical_nodes=num_canonical_nodes,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=pil_image_collate,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    device_transform_fn = NormalizationFn(
        mean=IMAGENET_CHANNEL_MEAN,
        std=IMAGENET_CHANNEL_STD,
        ignore_background=ignore_background,
    )

    return DataSpec(dataloader=dataloader, device_transforms=device_transform_fn)


def build_synthetic_ade20k_dataloader(
    global_batch_size: int,
    *,
    split: str = 'train',
    drop_last: bool = True,
    shuffle: bool = True,
    final_size: int = 512,
    num_unique_samples: int = 100,
    device: str = 'cpu',
    memory_format: MemoryFormat = MemoryFormat.CONTIGUOUS_FORMAT,
    **dataloader_kwargs: Dict[str, Any],
):
    """Builds a synthetic ADE20k dataloader.

    Args:
        batch_size (int): Global batch size.
        split (str): The dataset split to use either 'train', 'val', or 'test'. Default: ``'train```.
        drop_last (bool): Whether to drop last samples. Default: ``True``.
        shuffle (bool): Whether to shuffle the dataset. Default: ``True``.
        final_size (int): The final size of the image and target. Default: ``512``.
        num_unique_samples (int): Number of unique samples in synthetic dataset. Default: ``100``.
        device (str): Device with which to load the dataset. Default: ``cpu``.
        memory_format (:class:`composer.core.MemoryFormat`): Memory format of the tensors. Default: ``CONTIGUOUS_FORMAT``.
        **dataloader_kwargs (Dict[str, Any]): Additional settings for the dataloader (e.g. num_workers, etc.)
    """
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f'global_batch_size ({global_batch_size}) must be divisible by world_size ({dist.get_world_size()}).')
    batch_size = global_batch_size // dist.get_world_size()
    if split == 'train':
        total_dataset_size = 20_206
    elif split == 'val':
        total_dataset_size = 2_000
    else:
        total_dataset_size = 3_352

    dataset = SyntheticBatchPairDataset(
        total_dataset_size=total_dataset_size,
        data_shape=[3, final_size, final_size],
        label_shape=[final_size, final_size],
        num_classes=150,
        num_unique_samples_to_create=num_unique_samples,
        device=device,
        memory_format=memory_format,
    )
    sampler = dist.get_sampler(dataset, drop_last=drop_last, shuffle=shuffle)

    return DataSpec(
        DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            **dataloader_kwargs,
        ))


class RandomResizePair(torch.nn.Module):
    """Resize the image and target to ``base_size`` scaled by a randomly sampled value.

    Args:
        min_scale (float): the minimum value the samples can be rescaled.
        max_scale (float): the maximum value the samples can be rescaled.
        base_size (Tuple[int, int]): a specified base size (height x width) to scale to get the resized dimensions.
            When this is None, use the input image size. Default: ``None``.
    """

    def __init__(self, min_scale: float, max_scale: float, base_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.base_size = base_size

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        resize_scale = np.random.random_sample() * (self.max_scale - self.min_scale) + self.min_scale
        base_height, base_width = self.base_size if self.base_size else (image.height, image.width)
        resized_dims = (int(base_height * resize_scale), int(base_width * resize_scale))
        resized_image = TF.resize(image, resized_dims, interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        resized_target = TF.resize(target, resized_dims, interpolation=TF.InterpolationMode.NEAREST)  # type: ignore
        return resized_image, resized_target


# Based on: https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L584
class RandomCropPair(torch.nn.Module):
    """Crop the image and target at a randomly sampled position.

    Args:
        crop_size (Tuple[int, int]): the size (height x width) of the crop.
        class_max_percent (float): the maximum percent of the image area a single class should occupy. Default is 1.0.
        num_retry (int): the number of times to resample the crop if ``class_max_percent`` threshold is not reached.
            Default is 1.
    """

    def __init__(self, crop_size: Tuple[int, int], class_max_percent: float = 1.0, num_retry: int = 1):
        super().__init__()
        self.crop_size = crop_size
        self.class_max_percent = class_max_percent
        self.num_retry = num_retry

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample

        # if image size is smaller than crop size, no cropping necessary
        if image.height <= self.crop_size[0] and image.width <= self.crop_size[1]:
            return image, target

        # generate crop
        crop = transforms.RandomCrop.get_params(
            image, output_size=self.crop_size)  # type: ignore - transform typing excludes PIL.Image

        if self.class_max_percent < 1.0:
            for _ in range(self.num_retry):
                # Crop target
                target_crop = TF.crop(target, *crop)  # type: ignore - transform typing excludes PIL.Image

                # count the number of each class represented in cropped target
                labels, counts = np.unique(np.array(target_crop), return_counts=True)
                counts = counts[labels != 0]

                # if the class with the most area is within the class_max_percent threshold, stop retrying
                if len(counts) > 1 and (np.max(counts) / np.sum(counts)) < self.class_max_percent:
                    break

                crop = transforms.RandomCrop.get_params(
                    image, output_size=self.crop_size)  # type: ignore - transform typing excludes PIL.Image

        image = TF.crop(image, *crop)  # type: ignore - transform typing excludes PIL.Image
        target = TF.crop(target, *crop)  # type: ignore - transform typing excludes PIL.Image

        return image, target


class RandomHFlipPair(torch.nn.Module):
    """Flip the image and target horizontally with a specified probability.

    Args:
        probability (float): the probability of flipping the image and target. Default: ``0.5``.
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        if np.random.random_sample() > self.probability:
            image = TF.hflip(image)  # type: ignore - transform typing does not include PIL.Image
            target = TF.hflip(target)  # type: ignore - transform typing does not include PIL.Image
        return image, target


class PadToSize(torch.nn.Module):
    """Pad an image to a specified size.

    Args:
        size (Tuple[int, int]): the size (height x width) of the image after padding.
        fill (Union[int, Tuple[int, int, int]]): the value to use for the padded pixels. Default: ``0``.
    """

    def __init__(self, size: Tuple[int, int], fill: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, image: Image.Image):
        padding = max(self.size[0] - image.height, 0), max(self.size[1] - image.width, 0)
        padding = (padding[1] // 2, padding[0] // 2, ceil(padding[1] / 2), ceil(padding[0] / 2))
        image = TF.pad(image, padding, fill=self.fill)  # type: ignore - transform typing does not include PIL.Image
        return image


class PhotometricDistoration(torch.nn.Module):
    """Applies a combination of brightness, contrast, saturation, and hue jitters with random intensity.

    This is a less severe form of PyTorch's ColorJitter used by the mmsegmentation library here:
    https://github.com/open-mmlab/mmsegmentation/blob/aa50358c71fe9c4cccdd2abe42433bdf702e757b/mmseg/datasets/pipelines/transforms.py#L861

    Args:
        brightness (float): max and min to jitter brightness.
        contrast (float): max and min to jitter contrast.
        saturation (float): max and min to jitter saturation.
        hue (float): max and min to jitter hue.
    """

    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(self, image: Image.Image):
        if np.random.randint(2):
            brightness_factor = np.random.uniform(1 - self.brightness, 1 + self.brightness)
            image = TF.adjust_brightness(
                image, brightness_factor)  # type: ignore - transform typing does not include PIL.Image

        contrast_mode = np.random.randint(2)
        if contrast_mode == 1 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(
                image,  # type: ignore - transform typing does not include PIL.Image
                contrast_factor)

        if np.random.randint(2):
            saturation_factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
            image = TF.adjust_saturation(
                image, saturation_factor)  # type: ignore - transform typing does not include PIL.Image

        if np.random.randint(2):
            hue_factor = np.random.uniform(-self.hue, self.hue)
            image = TF.adjust_hue(image, hue_factor)  # type: ignore - transform typing does not include PIL.Image

        if contrast_mode == 0 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(
                image,  # type: ignore - transform typing does not include PIL.Image
                contrast_factor)

        return image


class ADE20k(Dataset):
    """PyTorch Dataset for ADE20k.

    Args:
        datadir (str): the path to the ADE20k folder.
        split (str): the dataset split to use, either 'training', 'validation', or 'test'. Default: ``'training'``.
        both_transforms (torch.nn.Module): transformations to apply to the image and target simultaneously.
            Default: ``None``.
        image_transforms (torch.nn.Module): transformations to apply to the image only. Default: ``None``.
        target_transforms (torch.nn.Module): transformations to apply to the target only. Default ``None``.
    """

    def __init__(self,
                 datadir: str,
                 split: str = 'training',
                 both_transforms: Optional[torch.nn.Module] = None,
                 image_transforms: Optional[torch.nn.Module] = None,
                 target_transforms: Optional[torch.nn.Module] = None):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.both_transforms = both_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

        # Check datadir value
        if self.datadir is None:
            raise ValueError('datadir must be specified')
        elif not os.path.exists(self.datadir):
            raise FileNotFoundError(f'datadir path does not exist: {self.datadir}')

        # Check split value
        if self.split not in ['training', 'validation', 'test']:
            raise ValueError(f'split must be one of [`training`, `validation`, `test`] but is: {self.split}')

        self.image_dir = os.path.join(self.datadir, 'images', self.split)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'ADE20k directory structure is not as expected: {self.image_dir} does not exist')

        self.image_files = os.listdir(self.image_dir)

        # Filter for ADE files
        self.image_files = [f for f in self.image_files if f[:3] == 'ADE']

        # Remove grayscale samples
        if self.split == 'training':
            corrupted_samples = ['00003020', '00001701', '00013508', '00008455']
            for sample in corrupted_samples:
                sample_file = f'ADE_train_{sample}.jpg'
                if sample_file in self.image_files:
                    self.image_files.remove(sample_file)

    def __getitem__(self, index):
        # Load image
        image_file = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path)

        # Load annotation target if using either train or val splits
        if self.split in ['training', 'validation']:
            target_path = os.path.join(self.datadir, 'annotations', self.split, image_file.split('.')[0] + '.png')
            target = Image.open(target_path)

            if self.both_transforms:
                image, target = self.both_transforms((image, target))

            if self.target_transforms:
                target = self.target_transforms(target)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.split in ['training', 'validation']:
            return image, target  # type: ignore
        else:
            return image

    def __len__(self):
        return len(self.image_files)
