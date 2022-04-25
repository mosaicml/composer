import copy
import pickle

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
    copied_model = copy.deepcopy(trainer.state.model)
    assert copied_model.logger is model.logger
    assert model.num_channels == copied_model.num_channels


def test_model_copy(dummy_train_dataloader: DataLoader):
    model = SimpleBatchPairModel(num_channels=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration="1ep", train_dataloader=dummy_train_dataloader)
    assert model.logger is not None
    copied_model = copy.copy(trainer.state.model)
    assert copied_model.logger is model.logger
    assert model.num_channels == copied_model.num_channels


def test_model_pickle(dummy_train_dataloader: DataLoader):
    model = SimpleBatchPairModel(num_channels=1, num_classes=1)
    assert model.logger is None
    trainer = Trainer(model=model, max_duration="1ep", train_dataloader=dummy_train_dataloader)
    assert model.logger is not None
    pickled_model = pickle.dumps(trainer.state.model)
    restored_model = pickle.loads(pickled_model)
    # after pickling the model, the restored loggers should be None, since the logger cannot be serialized
    assert restored_model.logger is None
    assert model.num_channels == restored_model.num_channels
