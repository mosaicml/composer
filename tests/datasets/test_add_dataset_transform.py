# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torchvision import transforms

from composer.datasets.synthetic import SyntheticPILDataset
from composer.datasets.utils import add_vision_dataset_transform

image_size = 32


def generate_synthetic_dataset(data_transforms):
    return SyntheticPILDataset(total_dataset_size=1000,
                               data_shape=[image_size, image_size],
                               num_classes=2,
                               transform=data_transforms)


def generate_default_transforms():
    return transforms.Compose([transforms.RandomCrop(32), transforms.ToTensor(), transforms.RandomRotation(5)])


def generate_composition_no_tensor():
    return transforms.Compose(
        [transforms.RandomCrop(32),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(5)])


@pytest.mark.parametrize('is_tensor_transform,index', [(False, 1), (True, 2)])
def test_pre_post_to_tensor_compose(is_tensor_transform, index):
    dataset = generate_synthetic_dataset(generate_default_transforms())
    add_vision_dataset_transform(dataset, transforms.RandomAutocontrast(), is_tensor_transform=is_tensor_transform)
    assert dataset.transform is not None
    assert type(dataset.transform.transforms[index]) == transforms.RandomAutocontrast


@pytest.mark.parametrize('is_tensor_transform,index', [(False, 0), (True, 1)])
def test_pre_post_to_tensor(is_tensor_transform, index):
    dataset = generate_synthetic_dataset(transforms.ToTensor())
    add_vision_dataset_transform(dataset, transforms.RandomAutocontrast(), is_tensor_transform=is_tensor_transform)
    assert dataset.transform is not None
    assert type(dataset.transform.transforms[index]) == transforms.RandomAutocontrast


@pytest.mark.parametrize('data_transforms', [(generate_composition_no_tensor()), (transforms.RandomHorizontalFlip())])
def test_default_to_append(data_transforms):
    dataset = generate_synthetic_dataset(data_transforms)
    add_vision_dataset_transform(dataset, transforms.RandomAutocontrast())
    assert dataset.transform is not None
    assert type(dataset.transform.transforms[-1]) == transforms.RandomAutocontrast


def test_add_to_none_transform():
    dataset = generate_synthetic_dataset(None)
    add_vision_dataset_transform(dataset, transforms.RandomAutocontrast())
    assert type(dataset.transform) == transforms.RandomAutocontrast
