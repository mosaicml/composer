import os
from dataclasses import dataclass
from math import ceil
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yahp as hp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from composer.core.types import Batch, Tensor
from composer.datasets.hparams import DataloaderSpec, DatasetHparams
from composer.utils import ddp


class TransformationFn:
    """Normalizes input data and removes the background class from target data if desired.

    Args:
        ignore_background (bool): Whether or not to ignore the background class when calculating the training loss.
    """

    def __init__(self, ignore_background: bool = True) -> None:
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None
        self.ignore_background = ignore_background

    def __call__(self, batch: Batch):
        xs, ys = batch
        assert isinstance(xs, Tensor)
        assert isinstance(ys, Tensor)
        device = xs.device

        if self.mean is None:
            self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=device)
            self.mean = self.mean.view(1, 3, 1, 1)
        if self.std is None:
            self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=device)
            self.std = self.std.view(1, 3, 1, 1)

        xs = xs.float()
        xs = xs.sub_(self.mean).div_(self.std)
        if self.ignore_background:
            ys = ys.sub_(1)
        return xs, ys


def fast_collate(batch: List[Tuple[Image.Image, Tensor]], memory_format: torch.memory_format = torch.contiguous_format):
    """Constructs a batch for training from individual samples.
    """
    imgs = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    image_tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    target_tensor = torch.zeros((len(targets), h, w), dtype=torch.int64).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2).copy()
        if nump_array.shape[0] != 3:
            assert nump_array.shape[0] == 1, "unexpected shape"
            nump_array = np.resize(nump_array, (3, h, w))
        assert image_tensor.shape[1:] == nump_array.shape, "shape mismatch"

        image_tensor[i] += torch.from_numpy(nump_array)
        target_tensor[i] += torch.from_numpy(np.array(targets[i], dtype=np.int64))

    return image_tensor, target_tensor


class RandomResizePair(torch.nn.Module):
    """Randomly select the scale to increase a base size for resizing both the image and target.

    Args:
        min_scale (float): the minimum value the samples can be rescaled.
        max_scale (float): the maximum value the samples can be rescaled.
        base_size (Tuple[int, int]): a specified base size to scale for the resized dimensions.
    """

    def __init__(self, min_scale: float, max_scale: float, base_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.base_size = base_size

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        resize_scale = np.random.random_sample() * (self.max_scale - self.min_scale) + self.min_scale
        width, height = self.base_size if self.base_size else image.size
        resized_dims = (int(height * resize_scale), int(width * resize_scale))
        resized_image = TF.resize(image, resized_dims, interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        resized_target = TF.resize(target, resized_dims, interpolation=TF.InterpolationMode.NEAREST)  # type: ignore
        return resized_image, resized_target


class RandomCropPair(torch.nn.Module):
    """Randomly position a fixed crop for both the image and target.

    Args:
        crop_size (Tuple[int, int]): the size of the image and target after cropping.
    """

    def __init__(self, crop_size: Tuple[int, int]):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        if image.width > self.crop_size[0] or image.height > self.crop_size[1]:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)  # type: ignore
            image = TF.crop(image, i, j, h, w)  # type: ignore
            target = TF.crop(target, i, j, h, w)  # type: ignore
        return image, target


class RandomHFlipPair(torch.nn.Module):
    """Randomly flip the image and target horizontally with a specified probability.

    Args:
        probability (float): the probability of flipping the image and target.
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        if np.random.random_sample() > self.probability:
            image = TF.hflip(image)  # type: ignore
            target = TF.hflip(target)  # type: ignore
        return image, target


class PadToSize(torch.nn.Module):
    """Pad an image to a specified size.

    Args:
        size (Tuple[int, int]): the final size of the image after padding.
        fill (Union[int, Tuple[int, int, int]]): the value to use for the padded pixels.
    """

    def __init__(self, size: Tuple[int, int], fill: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, image: Image.Image):
        padding = max(self.size[0] - image.width, 0), max(self.size[1] - image.height, 0)
        padding = (padding[0] // 2, padding[1] // 2, ceil(padding[0] / 2), ceil(padding[1] / 2))
        image = TF.pad(image, padding, fill=self.fill)  # type: ignore
        return image


class PhotometricDistoration(torch.nn.Module):
    """Applies a combination of brightness, contrast, saturation, and hue jitters with random intensity.

    This is a less intense form of PyTorch's ColorJitter used by the mmsegmentation library here:
    https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/datasets/pipelines/transforms.py#L835

    Args:
        brightness (float): how much to jitter brightness.
        contrast (float): how much to jitter contrast.
        saturation (float): how much to jitter saturation.
        hue (float): how much to jitter hue.
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
            image = TF.adjust_brightness(image, brightness_factor)  # type: ignore

        contrast_mode = np.random.randint(2)
        if contrast_mode == 1 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(image, contrast_factor)  # type: ignore

        if np.random.randint(2):
            saturation_factor = np.random.uniform(1 - self.saturation, 1 + self.saturation)
            image = TF.adjust_saturation(image, saturation_factor)  # type: ignore

        if np.random.randint(2):
            hue_factor = np.random.uniform(-self.hue, self.hue)
            image = TF.adjust_hue(image, hue_factor)  # type: ignore

        if contrast_mode == 0 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(image, contrast_factor)  # type: ignore

        return image


class ADE20k(Dataset):
    """PyTorch Dataset for ADE20k.

    Args:
        datadir (str): the path to the ADE20k folder.
        split (str): the dataset split to use, either 'train', 'val', or 'test'.
        both_transforms (torch.nn.Module): transformations to apply to the image and target simultaneously.
        image_transforms (torch.nn.Module): transformations to apply to the image only.
        target_transforms (torch.nn.Module): transformations to apply to the target only.
    """

    def __init__(self, datadir: str, split: str, both_transforms: torch.nn.Module, image_transforms: torch.nn.Module,
                 target_transforms: torch.nn.Module):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.both_transforms = both_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms

        self.image_files = os.listdir(os.path.join(self.datadir, 'images', self.split))

        # Filter for ADE files
        self.image_files = [f for f in self.image_files if f[:3] == 'ADE']

        # Remove grayscale samples
        if self.split == 'train':
            corrupted_samples = ['00003020', '00001701', '00013508', '00008455']
            for sample in corrupted_samples:
                self.image_files.remove(f'ADE_train_{sample}.jpg')

    def __getitem__(self, index):
        # Load image
        image_file = self.image_files[index]
        image_path = os.path.join(self.datadir, 'images', self.split, image_file)
        image = Image.open(image_path)

        # Load annotation target if using either train or val splits
        if self.split in ['train', 'val']:
            target_path = os.path.join(self.datadir, 'annotations', self.split, image_file.split('.')[0] + '.png')
            target = Image.open(target_path)

            if self.both_transforms:
                image, target = self.both_transforms((image, target))

            if self.target_transforms:
                target = self.target_transforms(target)

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.split in ['train', 'val']:
            return image, target  # type: ignore
        else:
            return image

    def __len__(self):
        return len(self.image_files)


@dataclass
class ADE20kDatasetHparams(DatasetHparams):
    """ Defines an instance of the ADE20k dataset for semantic segmentation.

    Args:
        split (str): the dataset split to use, either train, val, or test.
        base_size (int): initial size of the image and target before other augmentations.
        min_resize_scale (float): the minimum value the samples can be rescaled.
        max_resize_scale (float): the maximum value the samples can be rescaled.
        crop_size (int): size of the image and target after cropping.
        ignore_background (bool): whether or not to include the background class in training loss.

    """

    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')
    base_size: int = hp.optional("Size of the image to apply random scaling to", default=512)
    min_resize_scale: float = hp.optional("Minimum scale that the image can be randomly scaled by", default=0.5)
    max_resize_scale: float = hp.optional("Maximum scale that the image can be randomly scaled by", default=2.0)
    crop_size: int = hp.optional("Size of image after cropping", default=512)
    ignore_background: bool = hp.optional("Whether or not to include the background class in training loss",
                                          default=True)

    def validate(self):
        if self.datadir is None:
            raise ValueError("datadir must specify the path to the ADE20k dataset.")

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"split value {self.split} must be one of ['train', 'val', 'test'].")

        if self.base_size <= 0:
            raise ValueError("base_size cannot be zero or negative.")

        if self.min_resize_scale <= 0:
            raise ValueError("min_resize_scale cannot be zero or negative")

        if self.max_resize_scale < self.min_resize_scale:
            raise ValueError("max_resize_scale cannot be less than min_resize_scale")

    def initialize_object(self, batch_size, dataloader_hparams) -> DataloaderSpec:
        self.validate()

        # Define data transformations based on data split
        if self.split == 'train':
            both_transforms = transforms.Compose([
                RandomResizePair(min_scale=self.min_resize_scale,
                                 max_scale=self.max_resize_scale,
                                 base_size=(self.base_size, self.base_size)),
                RandomCropPair((self.crop_size, self.crop_size)),
                RandomHFlipPair(),
            ])
            image_transforms = transforms.Compose([
                PhotometricDistoration(brightness=32. / 255, contrast=0.5, saturation=0.5, hue=18. / 255),
                PadToSize((self.crop_size, self.crop_size), fill=(int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
            ])

            target_transforms = PadToSize((self.crop_size, self.crop_size), fill=0)
        else:
            both_transforms = None
            image_transforms = transforms.Resize(size=(self.base_size, self.base_size),
                                                 interpolation=TF.InterpolationMode.BILINEAR)
            target_transforms = transforms.Resize(size=(self.base_size, self.base_size),
                                                  interpolation=TF.InterpolationMode.NEAREST)

        dataset = ADE20k(self.datadir, self.split, both_transforms, image_transforms, target_transforms)  # type: ignore
        sampler = ddp.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return DataloaderSpec(dataloader=dataloader_hparams.initialize_object(dataset=dataset,
                                                                              batch_size=batch_size,
                                                                              sampler=sampler,
                                                                              collate_fn=fast_collate,
                                                                              drop_last=self.drop_last),
                              device_transform_fn=TransformationFn(self.ignore_background))
