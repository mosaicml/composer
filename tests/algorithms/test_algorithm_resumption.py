# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pathlib
from typing import Any, Callable, Dict

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from composer import Trainer
from composer.algorithms.factorize.factorize import Factorize
from composer.algorithms.layer_freezing.layer_freezing import LayerFreezing
from composer.algorithms.sam.sam import SAM
from composer.algorithms.squeeze_excite.squeeze_excite import SqueezeExcite
from composer.algorithms.stochastic_depth.stochastic_depth import StochasticDepth
from composer.core.algorithm import Algorithm
from composer.models.base import ComposerModel
from tests.algorithms.algorithm_settings import get_algorithm_parametrization
from tests.common import deep_compare, device


@pytest.fixture
def model(request):
    # Copy the model, so other parameterization start with a fresh model
    return copy.deepcopy(request.param)


@pytest.mark.timeout(180)
@device('gpu')
@pytest.mark.parametrize(
    "save_interval,save_filename,resume_file,final_checkpoint",
    [
        ["1ep", "ep{epoch}-rank{rank}", "ep2-rank{rank}", "latest-rank{rank}"],  # symlinking
        ["1ep", "ep{epoch}-rank{rank}", "ep3-rank{rank}", "ep5-rank{rank}"],  # test save at epoch end
    ],
)
@pytest.mark.parametrize("alg_cls,alg_kwargs,model,dataset", get_algorithm_parametrization(), indirect=['model'])
def test_algorithm_resumption(
    device,
    save_interval: int,
    save_filename: str,
    resume_file: str,
    final_checkpoint: str,
    tmp_path: pathlib.Path,
    alg_cls: Callable[..., Algorithm],
    alg_kwargs: Dict[str, Any],
    model: ComposerModel,
    dataset: Dataset,
):
    folder1 = os.path.join(tmp_path, 'folder1')
    folder2 = os.path.join(tmp_path, 'folder2')

    copied_model = copy.deepcopy(model)  # copy the model so the params will start from the same point

    if alg_cls is LayerFreezing:
        pytest.xfail('Known issues')

    if alg_cls in (SAM, SqueezeExcite, StochasticDepth, Factorize):
        pytest.xfail('Incompatible with optimizers that store state, e.g. Adam.')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    shared_config = {
        'train_dataloader': DataLoader(dataset=dataset, batch_size=4),
        'max_duration': '5ep',
        'device': device,
        'save_filename': save_filename,
        'save_interval': save_interval,
        'train_subset_num_batches': 2,
    }

    # train model once, saving checkpoints every epoch
    trainer1 = Trainer(
        model=model,
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
        load_path=os.path.join(folder1, resume_file),
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
