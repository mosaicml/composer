# Copyright 2021 MosaicML. All Rights Reserved.

import pytest
from torchvision import transforms

from composer.datasets.synthetic import SyntheticBatchPairDataset
from composer.utils.data import add_dataset_transform

image_size = 32

transformation_lists = [
    transforms.Compose([transforms.RandomCrop(32),
                        transforms.ToTensor(),
                        transforms.RandomRotation(5)]),
    transforms.ToTensor(),
    transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomRotation(5)
    ]), None
]


def generate_synthetic_dataset(data_transforms):
    return SyntheticBatchPairDataset(total_dataset_size=1000,
                                     data_shape=[64, 3, image_size, image_size],
                                     num_classes=2,
                                     transform=data_transforms)


def generate_default_transforms():
    return transforms.Compose([transforms.RandomCrop(32), transforms.ToTensor(), transforms.RandomRotation(5)])


@pytest.fixture(params=["pre", "post"])
def pre_post(request):
    return request.param


@pytest.fixture(params=["start", "end", transforms.ToTensor])
def locations_duplicate(request):
    return request.param


@pytest.mark.parametrize("data_transforms", [
    transforms.Compose([transforms.RandomCrop(32),
                        transforms.ToTensor(),
                        transforms.RandomRotation(5)]),
    transforms.ToTensor(),
])
def test_duplicate_transform(data_transforms, locations_duplicate, pre_post):
    dataset = generate_synthetic_dataset(data_transforms)
    dataset = add_dataset_transform(dataset, transforms.ToTensor(), location=locations_duplicate, pre_post=pre_post)
    if type(dataset.transform) == transforms.Compose:  # type: ignore
        n_to_tensor = 0
        for t in dataset.transform.transforms:  # type: ignore
            if type(t) == transforms.ToTensor:
                n_to_tensor += 1
        assert n_to_tensor == 1
    else:
        assert type(dataset.transform) == transforms.ToTensor  # type: ignore


@pytest.mark.parametrize("pre_post,index", [("pre", 1), ("post", 2)])
def test_pre_post_transform(pre_post, index):
    dataset = generate_synthetic_dataset(generate_default_transforms())
    dataset = add_dataset_transform(dataset,
                                    transforms.RandomAutocontrast(),
                                    location=transforms.ToTensor,
                                    pre_post=pre_post)
    assert type(dataset.transform.transforms[index]) == transforms.RandomAutocontrast  # type: ignore


@pytest.mark.parametrize("location,index", [("start", 0), ("end", -1)])
def test_start_end(location, index):
    dataset = generate_synthetic_dataset(generate_default_transforms())
    dataset = add_dataset_transform(dataset, transforms.RandomAutocontrast(), location=location)
    assert type(dataset.transform.transforms[index]) == transforms.RandomAutocontrast  # type: ignore
