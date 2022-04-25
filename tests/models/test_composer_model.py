import copy

from composer.core.types import DataLoader
from composer.trainer import Trainer
from tests.fixtures.models import SimpleBatchPairModel


def test_model_access_to_logger(dummy_train_dataloader: DataLoader):
    model = SimpleBatchPairModel(num_channels=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration="1ep", train_dataloader=dummy_train_dataloader)
    assert model.logger is trainer.logger


def test_model_deepcopy(dummy_train_dataloader: DataLoader):
    model = SimpleBatchPairModel(num_channels=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration="1ep", train_dataloader=dummy_train_dataloader)
    assert model.logger is not None
    copy.deepcopy(trainer.state.model)
