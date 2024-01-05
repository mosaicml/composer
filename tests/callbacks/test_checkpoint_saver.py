# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.callbacks import CheckpointSaver
from composer.core import Timestamp


def test_stateful_checkpoint_saver():
    checkpoint_saver = CheckpointSaver()
    assert not checkpoint_saver.all_saved_checkpoints_to_timestamp

    empty_state_dict = checkpoint_saver.state_dict()
    assert 'all_saved_checkpoints_to_timestamp' in empty_state_dict
    assert len(empty_state_dict['all_saved_checkpoints_to_timestamp']) == 0

    new_state_dict = {
        'all_saved_checkpoints_to_timestamp': {
            'example-checkpoint.pt': Timestamp(epoch=1, batch=2),
        },
    }

    checkpoint_saver.load_state_dict(new_state_dict)
    assert checkpoint_saver.all_saved_checkpoints_to_timestamp
    assert 'example-checkpoint.pt' in checkpoint_saver.all_saved_checkpoints_to_timestamp
    ts = checkpoint_saver.all_saved_checkpoints_to_timestamp['example-checkpoint.pt']
    assert ts.epoch == 1
    assert ts.batch == 2
    assert ts.sample == 0

    state_dict = checkpoint_saver.state_dict()
    assert 'all_saved_checkpoints_to_timestamp' in state_dict
    assert len(state_dict['all_saved_checkpoints_to_timestamp']) == 1
    assert 'example-checkpoint.pt' in state_dict['all_saved_checkpoints_to_timestamp']
    ts = state_dict['all_saved_checkpoints_to_timestamp']['example-checkpoint.pt']
    assert ts.epoch == 1
    assert ts.batch == 2
    assert ts.sample == 0
