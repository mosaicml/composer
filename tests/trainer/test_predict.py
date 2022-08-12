# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib

import pytest
import torch
from torch.utils.data import DataLoader

from composer.core import Callback, Event, State
from composer.loggers import Logger, LogLevel
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
        assert expected == actual, f'Event {event} expected to be called {expected} times, but instead it was called {actual} times'


class PredictionSaver(Callback):

    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def predict_batch_end(self, state: State, logger: Logger) -> None:
        name = f'batch_{int(state.predict_timestamp.batch)}.pt'
        filepath = os.path.join(self.folder, name)
        torch.save(state.outputs, filepath)

        # Also log the outputs as an artifact
        logger.file_artifact(LogLevel.BATCH, artifact_name=name, file_path=filepath)


class TestTrainerPredict():

    @pytest.mark.parametrize('subset_num_batches', [-1, 1])
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

    def test_timestamps(self):
        # Construct the trainer
        event_counter_callback = EventCounterCallback()
        trainer = Trainer(
            model=SimpleModel(),
            callbacks=[event_counter_callback],
        )

        # Predict on the model
        predict_dataloader = DataLoader(dataset=RandomClassificationDataset())
        trainer.predict(predict_dataloader)

        # Ensure that the predict timestamp matches the number of prediction events
        assert event_counter_callback.event_to_num_calls[
            Event.PREDICT_BATCH_START] == trainer.state.predict_timestamp.batch
        assert trainer.state.predict_timestamp.batch == trainer.state.predict_timestamp.batch_in_epoch

        # Ensure that if we predict again, the predict timestamp was reset

        # Reset the event counter callback
        event_counter_callback.event_to_num_calls = {k: 0 for k in event_counter_callback.event_to_num_calls}

        # Predict again
        trainer.predict(predict_dataloader)

        # Validate the same invariants
        assert event_counter_callback.event_to_num_calls[
            Event.PREDICT_BATCH_START] == trainer.state.predict_timestamp.batch
        assert trainer.state.predict_timestamp.batch == trainer.state.predict_timestamp.batch_in_epoch

    @pytest.mark.parametrize('return_outputs', [True, False])
    @pytest.mark.parametrize('device', ['cpu', pytest.param('gpu', marks=pytest.mark.gpu)])
    def test_return_outputs(self, return_outputs: bool, tmp_path: pathlib.Path, device: str):
        # Construct the trainer
        folder = str(tmp_path / 'prediction_outputs')
        prediction_saver_callback = PredictionSaver(folder)
        trainer = Trainer(
            model=SimpleModel(),
            device=device,
            callbacks=[prediction_saver_callback],
        )

        # Predict on the model
        predict_dataloader = DataLoader(dataset=RandomClassificationDataset())
        outputs = trainer.predict(predict_dataloader, subset_num_batches=1, return_outputs=return_outputs)

        if return_outputs:
            assert len(outputs) > 0
        else:
            assert len(outputs) == 0

        for output in outputs:
            assert output.device.type == 'cpu'

        loaded_output = torch.load(os.path.join(folder, 'batch_1.pt'), map_location='cpu')
        assert loaded_output.shape == (1, 2)
