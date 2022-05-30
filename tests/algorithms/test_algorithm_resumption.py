# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pathlib
from typing import Type

import pytest
import torch
from torch.utils.data import DataLoader

from composer import Algorithm, Trainer
from composer.algorithms import SAM, BlurPool, Factorize, LayerFreezing, SqueezeExcite, StochasticDepth
from tests.algorithms.algorithm_settings import get_alg_dataset, get_alg_kwargs, get_alg_model, get_algs_with_marks
from tests.common import deep_compare


@pytest.mark.timeout(30)
@pytest.mark.gpu
@pytest.mark.parametrize("alg_cls", get_algs_with_marks())
def test_algorithm_resumption(
    tmp_path: pathlib.Path,
    alg_cls: Type[Algorithm],
):
    folder1 = os.path.join(tmp_path, 'folder1')
    folder2 = os.path.join(tmp_path, 'folder2')

    model = get_alg_model(alg_cls)
    alg_kwargs = get_alg_kwargs(alg_cls)

    copied_model = copy.deepcopy(model)  # copy the model so the params will start from the same point

    if alg_cls is LayerFreezing:
        pytest.xfail('Known issues')

    if alg_cls in (SAM, SqueezeExcite, StochasticDepth, Factorize, BlurPool):
        pytest.xfail('Incompatible with optimizers that store state, e.g. Adam.')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    shared_config = {
        'max_duration': '2ep',
        'save_filename': 'ep{epoch}-rank{rank}',
        'train_subset_num_batches': 2,
    }

    # train model once, saving checkpoints every epoch
    trainer1 = Trainer(
        model=model,
        train_dataloader=DataLoader(dataset=get_alg_dataset(alg_cls), batch_size=4),
        optimizers=optimizer,
        schedulers=scheduler,
        save_folder=folder1,
        algorithms=alg_cls(**alg_kwargs),
        **shared_config,
    )
    trainer1.fit()

    # create second trainer, load an intermediate checkpoint
    # and continue training

    optimizer = torch.optim.Adam(copied_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    trainer2 = Trainer(
        model=copied_model,
        train_dataloader=DataLoader(dataset=get_alg_dataset(alg_cls), batch_size=4),
        load_path=os.path.join(folder1, 'ep1-rank{rank}'),
        load_weights_only=False,
        load_strict_model_weights=False,
        optimizers=optimizer,
        schedulers=scheduler,
        save_folder=folder2,
        algorithms=alg_cls(**alg_kwargs),
        **shared_config,
    )
    trainer2.fit()

    # check that the checkpoints are equal
    _assert_checkpoints_equal(
        file1=os.path.join(folder1, "ep2-rank0"),
        file2=os.path.join(folder2, "ep2-rank0"),
    )

    # check that different epoch checkpoints are _not_ equal
    # this ensures that the model weights are being updated.
    with pytest.raises(AssertionError):
        _assert_model_weights_equal(
            file1=os.path.join(folder1, 'ep1-rank0'),
            file2=os.path.join(folder1, "ep2-rank0"),
        )


def _assert_checkpoints_equal(file1, file2):
    checkpoint1 = torch.load(file1)
    checkpoint2 = torch.load(file2)

    # compare rng
    deep_compare(checkpoint1['rng'], checkpoint2['rng'])

    # compare state
    # remove the wall clock time fields since they will always differ
    del checkpoint1['state']['timestamp']['Timestamp']['total_wct']
    del checkpoint1['state']['timestamp']['Timestamp']['epoch_wct']
    del checkpoint1['state']['timestamp']['Timestamp']['batch_wct']
    del checkpoint2['state']['timestamp']['Timestamp']['total_wct']
    del checkpoint2['state']['timestamp']['Timestamp']['epoch_wct']
    del checkpoint2['state']['timestamp']['Timestamp']['batch_wct']
    deep_compare(checkpoint1['state'], checkpoint2['state'])


def _assert_model_weights_equal(file1, file2):
    checkpoint1 = torch.load(file1)
    checkpoint2 = torch.load(file2)

    deep_compare(checkpoint1['state']['model'], checkpoint2['state']['model'])
