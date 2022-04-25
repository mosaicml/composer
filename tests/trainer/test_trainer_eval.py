import torchmetrics
from torch.utils.data import DataLoader

from composer.core import Event
from composer.trainer import Trainer
from tests.common import EventCounterCallback, RandomClassificationDataset, SimpleModel


def test_trainer_eval_only():
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,  # TODO(ravi): Remove in #948
        max_duration='1ep',  # TODO(ravi): Remove in #948
    )

    eval_dataloader = DataLoader(dataset=RandomClassificationDataset())

    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
    )

    # Assert that there is some accuracy
    assert trainer.state.current_metrics['eval']['Accuracy'] != 0.0


def test_trainer_eval_subset_num_batches():
    train_dataloader = DataLoader(dataset=RandomClassificationDataset())
    event_counter_callback = EventCounterCallback()
    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=train_dataloader,  # TODO(ravi): Remove in #948
        max_duration='1ep',  # TODO(ravi): Remove in #948
        callbacks=[event_counter_callback],
    )

    eval_dataloader = DataLoader(dataset=RandomClassificationDataset())

    trainer.eval(
        dataloader=eval_dataloader,
        dataloader_label='eval',
        metrics=torchmetrics.Accuracy(),
        subset_num_batches=1,
    )

    assert event_counter_callback.event_to_num_calls[Event.EVAL_START] == 1
    assert event_counter_callback.event_to_num_calls[Event.EVAL_BATCH_START] == 1
