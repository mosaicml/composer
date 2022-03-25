
from composer.datasets.utils import _default_split_batch

from typing import Dict, List, Tuple

import pytest
import torch


def dummy_tensor_batch() -> torch.Tensor:
    return torch.randn(size=(12, 3, 32, 32))


def dummy_tuple_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return image, target


def dummy_tuple_batch_long() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_1 = torch.randn(size=(12, 3, 32, 32))
    image_2 = torch.randn(size=(12, 3, 32, 32))
    image_3 = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return image_1, image_2, image_3, target


def dummy_dict_batch() -> Dict[str, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return {'image': image, 'target': target}


def dummy_maskrcnn_batch() -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

    def generate_maskrcnn_sample(num_detections, image_height=12, image_width=12, num_classes=80):
        """Generates a maskrcnn style sample: (Tensor, Dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {
            'boxes':
                torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
            'labels':
                torch.randint(size=(num_detections,), low=0, high=num_classes),
            'masks':
                torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8)
        }
        return image, target

    def generate_maskrcnn_batch(batch_size, max_detections):
        return [generate_maskrcnn_sample(n) for n in torch.randint(size=(batch_size,), low=1, high=max_detections)]

    return generate_maskrcnn_batch(batch_size=12, max_detections=5)

@pytest.mark.parametrize(
    "batch",
    [dummy_tensor_batch(),
     dummy_tuple_batch(),
     dummy_tuple_batch_long(),
     dummy_dict_batch(),
     dummy_maskrcnn_batch()])
def test_default_split_without_error(batch):
    _default_split_batch(batch, num_microbatches=3)


@pytest.mark.parametrize(
    "batch",
    [dummy_tensor_batch(),
     dummy_tuple_batch(),
     dummy_tuple_batch_long(),
     dummy_dict_batch(),
     dummy_maskrcnn_batch()])
def test_default_split_num_microbatches(batch):
    microbatch = _default_split_batch(batch, num_microbatches=3)
    print(len(microbatch))
    assert len(microbatch) == 3


