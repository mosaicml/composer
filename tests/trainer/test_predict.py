# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer.core.event import Event
from composer.trainer.trainer import Trainer
from tests.common import EventCounterCallback, RandomClassificationDataset, SimpleModel


def _assert_predict_events_called_expected_number_of_times(
    event_counter: EventCounterCallback,
    num_predict_steps: int,
    num_predicts: int = 1,
):
    event_to_num_expected_invocations = {
        Event.PREDICT_START: num_predicts,
        Event.PREDICT_BATCH_START: num_predict_steps,
        Event.PREDICT_BEFORE_FORWARD: num_predict_steps,
        Event.PREDICT_AFTER_FORWARD: num_predict_steps,
        Event.PREDICT_BATCH_END: num_predict_steps,
        Event.PREDICT_END: num_predicts,
    }

    for event, expected in event_to_num_expected_invocations.items():
        actual = event_counter.event_to_num_calls[event]
        assert expected == actual, f"Event {event} expected to be called {expected} times, but instead it was called {actual} times"


class TestTrainerPredict():

    @pytest.mark.parametrize("subset_num_batches", [-1, 1])
    def test_predict(self, subset_num_batches: int):
        # Create the trainer and train
        event_counter_callback = EventCounterCallback()
        trainer = Trainer(
            model=SimpleModel(),
            train_dataloader=DataLoader(dataset=RandomClassificationDataset()),
            max_duration='1ba',
            callbacks=[event_counter_callback],
        )
        trainer.fit()

        # Remove the datalaoder from the state (to ensure that the predict dl is being used)
        trainer.state.set_dataloader(None)

        # Run predict()
        predict_dl = DataLoader(dataset=RandomClassificationDataset())
        trainer.predict(predict_dl, subset_num_batches)

        # Validate that the predict events were called the correct number of times
        num_predict_batches = subset_num_batches if subset_num_batches >= 0 else len(predict_dl)
        _assert_predict_events_called_expected_number_of_times(event_counter_callback, num_predict_batches)
