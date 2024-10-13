# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from unittest import mock
from unittest.mock import call

from torch.utils.data import DataLoader

from composer.callbacks import LoadCheckpoint
from composer.core.state import State
from composer.models.huggingface import HuggingFaceModel
from composer.trainer.trainer import Trainer
from tests.common.datasets import RandomTextLMDataset


def test_load_checkpoint_callback(
    tiny_gpt2_model,
    tiny_gpt2_tokenizer,
    gpt2_peft_config,
):

    model = HuggingFaceModel(
        tiny_gpt2_model,
        tokenizer=tiny_gpt2_tokenizer,
        peft_config=gpt2_peft_config,
        should_save_peft_only=True,
    )

    # Function to check the arguments passed to the load_checkpoint function.
    def check_callback_load_args(state: State, **kwargs):
        assert state.model == model

        # Check that the `should_save_peft_only` flag on the model was set to False when loading the checkpoint.
        assert state.model.should_save_peft_only == False

    # Patch the load_checkpoint function to check the arguments passed to it.
    with mock.patch(
        'composer.callbacks.load_checkpoint.load_checkpoint',
        new=mock.MagicMock(wraps=check_callback_load_args),
    ) as callback_load:
        with mock.patch('composer.trainer.trainer.checkpoint.load_checkpoint') as trainer_load:

            calls = mock.MagicMock()
            calls.attach_mock(trainer_load, 'trainer_load')
            calls.attach_mock(callback_load, 'callback_load')

            Trainer(
                model=model,
                callbacks=[LoadCheckpoint(
                    load_path='fake-path',
                    event='BEFORE_LOAD',
                )],
                train_dataloader=DataLoader(RandomTextLMDataset()),
                max_duration='1ba',
                load_path='fake_path',
            )

            callback_load.assert_called_once()
            trainer_load.assert_called_once()

            # Assert that the callback_load and trainer_load functions were called in the correct order.
            assert calls.mock_calls == [
                call.callback_load(**callback_load.call_args.kwargs),
                call.trainer_load(**trainer_load.call_args.kwargs),
            ]

    # Check that the `should_save_peft_only` flag on the model was reset to its original value after loading the checkpoint.
    assert model.should_save_peft_only == True
