# Copyright 2021 MosaicML. All Rights Reserved.

from typing import Callable, Optional, Sequence, Union

import torch
import torch.utils.data
from PIL import Image
from torchvision.datasets import VisionDataset

from composer.core.types import MemoryFormat
from composer.utils.string_enum import StringEnum


class SyntheticDataType(StringEnum):
    GAUSSIAN = "gaussian"
    SEPARABLE = "separable"


class SyntheticDataLabelType(StringEnum):
    CLASSIFICATION_INT = "classification_int"
    CLASSIFICATION_ONE_HOT = "classification_one_hot"


class SyntheticBatchPairDataset(torch.utils.data.Dataset):
    """Emulates a dataset of provided size and shape.

    Args:
        total_dataset_size (int): The total size of the dataset to emulate.
        data_shape (List[int]): Shape of the tensor for input samples.
        num_unique_samples_to_create (int): The number of unique samples to allocate memory for.
        data_type (str or SyntheticDataType, optional), Type of synthetic data to create.
        label_type (str or SyntheticDataLabelType, optional), Type of synthetic data to create.
        num_classes (int, optional): Number of classes to use. Required if
            ``SyntheticDataLabelType`` is ``CLASSIFICATION_INT`` or``CLASSIFICATION_ONE_HOT``. Otherwise, should be ``None``.
        label_shape (List[int]): Shape of the tensor for each sample label.
        device (str): Device to store the sample pool. Set to ``cuda`` to store samples
            on the GPU and eliminate PCI-e bandwidth with the dataloader. Set to `cpu`
            to move data between host memory and the gpu on every batch.
        memory_format (MemoryFormat, optional): Memory format for the sample pool.
    """

    def __init__(self,
                 *,
                 total_dataset_size: int,
                 data_shape: Sequence[int],
                 num_unique_samples_to_create: int = 100,
                 data_type: Union[str, SyntheticDataType] = SyntheticDataType.GAUSSIAN,
                 label_type: Union[str, SyntheticDataLabelType] = SyntheticDataLabelType.CLASSIFICATION_INT,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 device: str = "cpu",
                 memory_format: Union[str, MemoryFormat] = MemoryFormat.CONTIGUOUS_FORMAT,
                 transform: Optional[Callable] = None):
        self.total_dataset_size = total_dataset_size
        self.data_shape = data_shape
        self.num_unique_samples_to_create = num_unique_samples_to_create
        self.data_type = SyntheticDataType(data_type)
        self.label_type = SyntheticDataLabelType(label_type)
        self.num_classes = num_classes
        self.label_shape = label_shape
        self.device = device
        self.memory_format = MemoryFormat(memory_format)
        self.transform = transform

        self._validate_label_inputs(label_type=self.label_type,
                                    num_classes=self.num_classes,
                                    label_shape=self.label_shape)

        # The synthetic data
        self.input_data = None
        self.input_target = None

    def _validate_label_inputs(self, label_type: SyntheticDataLabelType, num_classes: Optional[int],
                               label_shape: Optional[Sequence[int]]):
        if label_type == SyntheticDataLabelType.CLASSIFICATION_INT or label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
            if num_classes is None or num_classes <= 0:
                raise ValueError("classification label_types require num_classes > 0")

    def __len__(self) -> int:
        return self.total_dataset_size

    def __getitem__(self, idx: int):
        idx = idx % self.num_unique_samples_to_create
        if self.input_data is None:
            # Generating data on the first call to __getitem__ so that data is stored on the correct gpu,
            # after DeviceSingleGPU calls torch.cuda.set_device
            # This does mean that the first batch will be slower
            # generating samples so all values for the sample are the sample index
            # e.g. all(input_data[1] == 1). Helps with debugging.
            assert self.input_target is None
            input_data = torch.randn(self.num_unique_samples_to_create, *self.data_shape, device=self.device)

            input_data = torch.clone(input_data)  # allocate actual memory
            input_data = input_data.contiguous(memory_format=getattr(torch, self.memory_format.value))

            if self.label_type == SyntheticDataLabelType.CLASSIFICATION_ONE_HOT:
                assert self.num_classes is not None
                input_target = torch.zeros((self.num_unique_samples_to_create, self.num_classes), device=self.device)
                input_target[:, 0] = 1.0
            elif self.label_type == SyntheticDataLabelType.CLASSIFICATION_INT:
                assert self.num_classes is not None
                if self.label_shape:
                    label_batch_shape = (self.num_unique_samples_to_create, *self.label_shape)
                else:
                    label_batch_shape = (self.num_unique_samples_to_create,)
                input_target = torch.randint(0, self.num_classes, label_batch_shape, device=self.device)
            else:
                raise ValueError(f"Unsupported label type {self.data_type}")

            # If separable, force the positive examples to have a higher mean than the negative examples
            if self.data_type == SyntheticDataType.SEPARABLE:
                assert self.label_type == SyntheticDataLabelType.CLASSIFICATION_INT, \
                    "SyntheticDataType.SEPARABLE requires integer classes."
                assert max(input_target) == 1 and min(input_target) == 0, \
                    "SyntheticDataType.SEPARABLE only supports binary labels"
                # Make positive examples have mean = 3 and negative examples have mean = -3
                # so they are easier to separate with a classifier
                input_data[input_target == 0] -= 3
                input_data[input_target == 1] += 3

            self.input_data = input_data
            self.input_target = input_target

        assert self.input_target is not None

        if self.transform is not None:
            return self.transform(self.input_data[idx]), self.input_target[idx]
        else:
            return self.input_data[idx], self.input_target[idx]


class SyntheticPILDataset(VisionDataset):
    """Similar to :class:`SyntheticBatchPairDataset`, but yields samples of type :class:`~Image.Image` and supports
    dataset transformations.

    Args:
        total_dataset_size (int): The total size of the dataset to emulate.
        data_shape (List[int]): Shape of the image for input samples. Default = [64, 64]
        num_unique_samples_to_create (int): The number of unique samples to allocate memory for.
        data_type (str or SyntheticDataType, optional), Type of synthetic data to create.
        label_type (str or SyntheticDataLabelType, optional), Type of synthetic data to create.
        num_classes (int, optional): Number of classes to use. Required if
            ``SyntheticDataLabelType`` is ``CLASSIFICATION_INT`` or
            ``CLASSIFICATION_ONE_HOT``. Otherwise, should be ``None``.
        label_shape (List[int]): Shape of the tensor for each sample label.
        transform (Callable): Dataset transforms
    """

    def __init__(self,
                 *,
                 total_dataset_size: int,
                 data_shape: Sequence[int] = (64, 64, 3),
                 num_unique_samples_to_create: int = 100,
                 data_type: Union[str, SyntheticDataType] = SyntheticDataType.GAUSSIAN,
                 label_type: Union[str, SyntheticDataLabelType] = SyntheticDataLabelType.CLASSIFICATION_INT,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 transform: Optional[Callable] = None):
        super().__init__(root="", transform=transform)
        self._dataset = SyntheticBatchPairDataset(
            total_dataset_size=total_dataset_size,
            data_shape=data_shape,
            data_type=data_type,
            num_unique_samples_to_create=num_unique_samples_to_create,
            label_type=label_type,
            num_classes=num_classes,
            label_shape=label_shape,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        input_data, target = self._dataset[idx]

        input_data = input_data.numpy()

        # Shift and scale to be [0, 255]
        input_data = (input_data - input_data.min())
        input_data = (input_data * (255 / input_data.max())).astype("uint8")

        sample = Image.fromarray(input_data)
        if self.transform is not None:
            return self.transform(sample), target
        else:
            return sample, target
