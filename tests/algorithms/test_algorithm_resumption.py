# Copyright 2021 MosaicML. All Rights Reserved.

import os
from typing import Optional

import py
import pytest
import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer.algorithms import get_algorithm_registry
from tests.algorithms.algorithm_settings import get_settings
from tests.common import device
from tests.test_state import _check_dict_recursively
from tests.utils.deep_compare import deep_compare


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
@pytest.mark.parametrize("algorithm", get_algorithm_registry().keys())
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
    if algorithm in ('squeeze_excite', 'ghost_batchnorm', 'layer_freezing', 'blurpool'):
        pytest.xfail('Known issues.')

    setting = get_settings(algorithm)
    if setting is None:
        pytest.skip('No setting provided in algorithm_settings.')

    folder1 = os.path.join(tmpdir, 'folder1')
    folder2 = os.path.join(tmpdir, 'folder2')

    config = {
        'algorithms': setting['algorithm'],
        'model': setting['model'],
        'train_dataloader': DataLoader(dataset=setting['dataset'], batch_size=4),
        'max_duration': '5ep',
        'loggers': [],
        'seed': seed,
        'device': device,
        'deterministic_mode': True,
        'save_filename': save_filename,
        'save_folder': folder1,
        'save_interval': save_interval,
        'train_subset_num_batches': 5,
    }

    # train model once, saving checkpoints every epoch
    trainer1 = Trainer(**config)
    trainer1.fit()

    # create second trainer, load an intermediate checkpoint
    # and continue training
    config.update({
        'save_folder': folder2,
        'load_path': os.path.join(folder1, resume_file),
        'load_weights_only': False,
        'load_strict': False,
    })
    trainer2 = Trainer(**config)
    trainer2.fit()

    # check that the checkpoints are equal
    _assert_checkpoints_equal(
        file1=os.path.join(folder1, final_checkpoint.format(rank=0)),
        file2=os.path.join(folder2, final_checkpoint.format(rank=0)),
    )


def _assert_checkpoints_equal(file1, file2):
    checkpoint1 = torch.load(file1)
    checkpoint2 = torch.load(file2)

    # compare rng
    deep_compare(checkpoint1['rng'], checkpoint2['rng'])

    # compare state
    _check_dict_recursively(checkpoint1['state'], checkpoint2['state'])
