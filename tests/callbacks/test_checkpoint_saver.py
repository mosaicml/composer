# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from composer.callbacks import CheckpointSaver
from composer.core import Timestamp


def test_stateful_checkpoint_saver():
    checkpoint_saver = CheckpointSaver()
    assert not checkpoint_saver.all_saved_checkpoints_to_timestamp

    # empty state dict
    empty_state_dict = checkpoint_saver.state_dict()
    assert 'all_saved_checkpoints_to_timestamp' in empty_state_dict
    assert len(empty_state_dict['all_saved_checkpoints_to_timestamp']) == 0

    # backwards compatibility; empty state dict should not raise
    checkpoint_saver.load_state_dict({})
    assert not checkpoint_saver.all_saved_checkpoints_to_timestamp

    # add a checkpoint and confirm it can save and load
    checkpoint_saver.all_saved_checkpoints_to_timestamp = {
        'foobar/example-checkpoint.pt': Timestamp(epoch=1, batch=2),
    }
    new_state_dict = checkpoint_saver.state_dict()
    assert 'all_saved_checkpoints_to_timestamp' in new_state_dict
    assert len(new_state_dict['all_saved_checkpoints_to_timestamp']) == 1
    checkpoint, ts = new_state_dict['all_saved_checkpoints_to_timestamp'][0]
    assert checkpoint == 'foobar/example-checkpoint.pt'
    assert isinstance(ts, dict)
    assert ts['epoch'] == 1
    assert ts['batch'] == 2
    assert ts['sample'] == 0

    # load works again if we clear the dict
    checkpoint_saver.all_saved_checkpoints_to_timestamp = {}
    checkpoint_saver.load_state_dict(new_state_dict)
    assert checkpoint_saver.all_saved_checkpoints_to_timestamp
    assert len(checkpoint_saver.all_saved_checkpoints_to_timestamp) == 1
    assert 'foobar/example-checkpoint.pt' in checkpoint_saver.all_saved_checkpoints_to_timestamp
    ts = checkpoint_saver.all_saved_checkpoints_to_timestamp['foobar/example-checkpoint.pt']
    assert isinstance(ts, Timestamp)
    assert ts.epoch == 1
    assert ts.batch == 2
    assert ts.sample == 0
