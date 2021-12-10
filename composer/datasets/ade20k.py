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

    def __init__(self) -> None:
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None

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
        ys = ys.sub_(1)
        return xs, ys


def fast_collate(batch: List[Tuple[Image.Image, Tensor]], memory_format: torch.memory_format = torch.contiguous_format):
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


class ResizePair(torch.nn.Module):

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        image = TF.resize(image, (self.size, self.size), interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        target = TF.resize(target, (self.size, self.size), interpolation=TF.InterpolationMode.NEAREST)  # type: ignore
        return image, target


class RandomResizePair(torch.nn.Module):

    def __init__(self, min_scale: float, max_scale: float):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        resize_ratio = np.random.random_sample() * (self.max_scale - self.min_scale) + self.min_scale
        resized_dims = (int(image.width * resize_ratio), int(image.height * resize_ratio))
        resized_image = TF.resize(image, resized_dims, interpolation=TF.InterpolationMode.BILINEAR)  # type: ignore
        resized_target = TF.resize(target, resized_dims, interpolation=TF.InterpolationMode.NEAREST)  # type: ignore
        return resized_image, resized_target


class RandomCropPair(torch.nn.Module):

    def __init__(self, crop_size: int):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, sample: Tuple[Image.Image, Image.Image]):
        image, target = sample
        if image.width > self.crop_size or image.height > self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)  # type: ignore
            target = TF.crop(target, i, j, h, w)  # type: ignore
        return image, target


class RandomHFlipPair(torch.nn.Module):

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

    def __init__(self, size: int, fill: Union[int, Tuple[int, int, int]] = 0):
        super().__init__()
        self.size = size
        self.fill = fill

    def forward(self, image: Image.Image):
        padding = max(self.size - image.width, 0), max(self.size - image.height, 0)
        padding = (padding[0] // 2, padding[1] // 2, ceil(padding[0] / 2), ceil(padding[1] / 2))
        image = TF.pad(image, padding, fill=self.fill)  # type: ignore
        return image


class PhotometricDistoration(torch.nn.Module):

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
            hue_factor = np.random.uniform(self.hue, self.hue)
            image = TF.adjust_hue(image, hue_factor)  # type: ignore

        if contrast_mode == 0 and np.random.randint(2):
            contrast_factor = np.random.uniform(1 - self.contrast, 1 + self.contrast)
            image = TF.adjust_contrast(image, contrast_factor)  # type: ignore

        return image


class PILToMask:

    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64)


class ADE20k(Dataset):

    def __init__(self,
                 datadir: str,
                 split: str,
                 both_transforms: torch.nn.Module,
                 image_transforms: torch.nn.Module,
                 target_transforms: torch.nn.Module,
                 ignore_class: int = -1):
        super().__init__()
        self.datadir = datadir
        self.split = split
        self.both_transforms = both_transforms
        self.image_transforms = image_transforms
        self.target_transforms = target_transforms
        self.ignore_class = ignore_class

        self.image_files = os.listdir(os.path.join(self.datadir, 'images', self.split))

        # Grab only the ADE files
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

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.target_transforms:
            target = self.target_transforms(target)

        return image, target

    def __len__(self):
        return len(self.image_files)


@dataclass
class ADE20kDatasetHparams(DatasetHparams):

    split: str = hp.optional("Which split of the dataset to use. Either ['train', 'val', 'test']", default='train')
    resize_size: int = hp.optional("Size of the image after resizing", default=512)
    min_resize_scale: float = hp.optional("Minimum scale that the image can be randomly scaled by", default=0.5)
    max_resize_scale: float = hp.optional("Maximum scale that the image can be randomly scaled by", default=2.0)
    crop_size: int = hp.optional("Size of image after cropping", default=512)
    ignore_class: int = hp.optional("The integer to assign the ignore class", default=-1)

    def initialize_object(self, batch_size, dataloader_hparams) -> DataloaderSpec:
        both_transforms = [ResizePair(size=self.resize_size)]

        if self.split == 'train':
            both_transforms += [
                RandomResizePair(self.min_resize_scale, self.max_resize_scale),
                RandomCropPair(self.crop_size),
            ]
            both_transforms = transforms.Compose(both_transforms)
            image_transforms = transforms.Compose([
                PhotometricDistoration(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.0703125),
                PadToSize(self.crop_size, fill=(int(0.485 * 255), int(0.456 * 255), int(0.406 * 255)))
            ])
            target_transforms = transforms.Compose([PadToSize(self.crop_size, fill=self.ignore_class), PILToMask()])
        else:
            both_transforms = transforms.Compose(both_transforms)
            image_transforms = None
            target_transforms = PILToMask()
        dataset = ADE20k(self.datadir, self.split, both_transforms, image_transforms, target_transforms,
                         self.ignore_class)
        sampler = ddp.get_sampler(dataset, drop_last=self.drop_last, shuffle=self.shuffle)
        return DataloaderSpec(dataloader=dataloader_hparams.initialize_object(dataset=dataset,
                                                                              batch_size=batch_size,
                                                                              sampler=sampler,
                                                                              collate_fn=fast_collate,
                                                                              drop_last=self.drop_last),
                              device_transform_fn=TransformationFn())

    def validate(self):
        pass
