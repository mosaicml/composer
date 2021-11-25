# Copyright 2021 MosaicML. All Rights Reserved.

from torchvision import transforms


def add_dataset_transform(dataset, transform):
    """Flexibly add a transform to the dataset's collection of transforms.

    Args:
        dataset: A torchvision-like dataset
        transform: Function to be added to the dataset's collection of transforms

    Returns:
        The original dataset. The transform is added in-place.
    """

    if not hasattr(dataset, "transform"):
        raise ValueError(f"Dataset of type {type(dataset)} has no attribute 'transform'. Expected TorchVision dataset.")

    if dataset.transform is None:
        dataset.transform = transform
    elif hasattr(dataset.transform, "transforms"):  # transform is a Compose
        dataset.transform.transforms.append(transform)
    else:  # transform is some other basic transform, join using Compose
        dataset.transform = transforms.Compose([dataset.transform, transform])

    return dataset


def get_num_batches(num_samples: int, batch_size: int, drop_last: bool) -> int:
    if drop_last:
        return num_samples // batch_size
    else:
        return (num_samples - 1) // batch_size + 1
