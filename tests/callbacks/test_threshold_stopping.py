from composer import Trainer
from tests.common import SimpleModel
from tests.common.datasets import RandomClassificationDataset
from torch.utils.data import DataLoader
from composer.callbacks.threshold_stopping import ThresholdStopper


def test_threshold_stops():
    tstop = ThresholdStopper()

    trainer = Trainer(
        model=SimpleModel(num_features=5),
        train_dataloader=DataLoader(
            RandomClassificationDataset(shape=(5, 1, 1)),
            batch_size=4,
        ),
        max_duration="3ep",
        callbacks=[tstop],
    )

    trainer.fit()

    assert trainer.state.timer.epoch.value == 2