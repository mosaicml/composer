# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import py
import pytest
import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer.algorithms import get_algorithm_registry
from tests.algorithms.algorithm_settings import get_settings
from tests.common import deep_compare, device

ALGORITHMS = get_algorithm_registry().keys()


@pytest.mark.timeout(180)
@device('gpu')
@pytest.mark.parametrize(
    "seed,save_interval,save_filename,resume_file,final_checkpoint",
    [
        [None, "1ep", "ep{epoch}-rank{rank}", "ep2-rank{rank}", "latest-rank{rank}"
        ],  # test randomized seed saving and symlinking
        [42, "1ep", "ep{epoch}-rank{rank}", "ep3-rank{rank}", "ep5-rank{rank}"],  # test save at epoch end
    ],
)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_resumption(
    algorithm: str,
    device,
    seed: Optional[int],
    save_interval: int,
    save_filename: str,
    resume_file: str,
    final_checkpoint: str,
    tmpdir: py.path.local,
):
    if algorithm in ('no_op_model', 'scale_schedule'):
        pytest.skip('stub algorithms')

    if algorithm in ('cutmix, mixup, label_smoothing'):
        # see: https://github.com/mosaicml/composer/issues/362
        pytest.importorskip("torch", minversion="1.10", reason="Pytorch 1.10 required.")

    if algorithm in ('layer_freezing'):
        pytest.xfail('Known issues')

    if algorithm in ('sam', 'squeeze_excite', 'stochastic_depth', 'factorize'):
        pytest.xfail('Incompatible with optimizers that store state, e.g. Adam.')

    setting = get_settings(algorithm)
    if setting is None:
        pytest.xfail('No setting provided in algorithm_settings.')

    folder1 = os.path.join(tmpdir, 'folder1')
    folder2 = os.path.join(tmpdir, 'folder2')

    model = setting['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    config = {
        'algorithms': setting['algorithm'],
        'model': model,
        'train_dataloader': DataLoader(dataset=setting['dataset'], batch_size=4),
        'max_duration': '5ep',
        'device': device,
        'save_filename': save_filename,
        'save_folder': folder1,
        'save_interval': save_interval,
        'train_subset_num_batches': 2,
        'optimizers': optimizer,
        'schedulers': scheduler
    }

    # train model once, saving checkpoints every epoch
    trainer1 = Trainer(**config)
    trainer1.fit()

    # create second trainer, load an intermediate checkpoint
    # and continue training
    setting = get_settings(algorithm)
    assert setting is not None

    model = setting['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    config.update({
        'model': model,
        'save_folder': folder2,
        'load_path': os.path.join(folder1, resume_file),
        'load_weights_only': False,
        'load_strict': False,
        'optimizers': optimizer,
        'schedulers': scheduler,
    })
    trainer2 = Trainer(**config)
    trainer2.fit()

    # check that the checkpoints are equal
    _assert_checkpoints_equal(
        file1=os.path.join(folder1, final_checkpoint.format(rank=0)),
        file2=os.path.join(folder2, final_checkpoint.format(rank=0)),
    )

    # check that different epoch checkpoints are _not_ equal
    # this ensures that the model weights are being updated.
    with pytest.raises(AssertionError):
        _assert_model_weights_equal(
            file1=os.path.join(folder1, save_filename.format(epoch=4, rank=0)),
            file2=os.path.join(folder1, final_checkpoint.format(rank=0)),
        )


def _assert_checkpoints_equal(file1, file2):
    checkpoint1 = torch.load(file1)
    checkpoint2 = torch.load(file2)

    # compare rng
    deep_compare(checkpoint1['rng'], checkpoint2['rng'])

    # compare state
    deep_compare(checkpoint1['state'], checkpoint2['state'])


def _assert_model_weights_equal(file1, file2):
    checkpoint1 = torch.load(file1)
    checkpoint2 = torch.load(file2)

    deep_compare(checkpoint1['state']['model'], checkpoint2['state']['model'])
