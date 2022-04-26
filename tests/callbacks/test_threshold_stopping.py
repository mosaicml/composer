from composer import Trainer
from common import SimpleModel
from tests.common.datasets import RandomClassificationDataset
from composer.callbacks import ThresholdStopping


def test_threshold_stops():
    tstop = ThresholdStopping()

    trainer = Trainer(
        model=SimpleModel(),
        train_dataloader=RandomClassificationDataset(),
        max_duration="3ep",
        callbacks=[tstop],
    )

    trainer.fit()

    assert trainer.state.timer.epoch.value == 2