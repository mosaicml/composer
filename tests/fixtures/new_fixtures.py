# Copyright 2021 MosaicML. All Rights Reserved.

"""These fixtures are shared globally across the test suite."""
import pytest
from typing import Tuple, Dict, List

from torch.utils.data import DataLoader
import torch

from composer.core import Logger, State
from tests.common import RandomClassificationDataset, SimpleModel


@pytest.fixture
def minimal_state():
    """Most minimally defined state possible.

    Tests should configure the state for their specific needs.
    """
    return State(
        model=SimpleModel(),
        rank_zero_seed=0,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        evaluators=[],
        max_duration='100ep',
    )


@pytest.fixture
def empty_logger(minimal_state: State) -> Logger:
    """Logger without any output configured."""
    return Logger(state=minimal_state, backends=[])


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")

    yield


def dummy_tensor_batch() -> torch.Tensor:
    return torch.randn(size=(12, 3, 32, 32))


def dummy_tuple_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return image, target


def dummy_dict_batch() -> Dict[str, torch.Tensor]:
    image = torch.randn(size=(12, 3, 32, 32))
    target = torch.randint(size=(12,), high=10)
    return {'image': image, 'target': target}


def dummy_maskrcnn_batch() -> List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    def generate_maskrcnn_sample(num_detections, image_height=12, image_width=12, num_classes=80):
        """Generates a maskrcnn style sample (Tensor, Dict[Tensor])."""
        image = torch.randn(size=(3, image_height, image_width)).type(torch.float)
        target = {'boxes': torch.randint(size=(num_detections, 4), low=0, high=min(image_height, image_width)).type(torch.float),
                  'labels': torch.randint(size=(num_detections,), low=0, high=num_classes),
                  'masks': torch.randint(size=(num_detections, image_height, image_width), low=0, high=2).type(torch.uint8)}
        return image, target

    def generate_maskrcnn_batch(batch_size, max_detections):
        return [generate_maskrcnn_sample(n) for n in torch.randint(size=(batch_size,), low=1, high=max_detections)]

    return generate_maskrcnn_batch(batch_size=12, max_detections=5)
