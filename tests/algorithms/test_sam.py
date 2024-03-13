# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from composer import Trainer
from composer.algorithms import SAM
from composer.loss import soft_cross_entropy
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel


class TestSAMLossDict():

    @pytest.fixture
    def config(self):
        train_dataset = RandomClassificationDataset(size=16)

        return {
            'algorithms':
                SAM(),
            'model':
                SimpleModel(),
            'train_dataloader':
                DataLoader(
                    dataset=train_dataset,
                    batch_size=4,
                    sampler=dist.get_sampler(train_dataset),
                ),
            'max_duration':
                '2ep',
            'precision':
                'fp32',
        }

    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_sam_dict_loss(self, config):

        def dict_loss(outputs, targets, *args, **kwargs):
            losses = {}
            losses['cross_entropy1'] = 0.25 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            losses['cross_entropy2'] = 0.75 * soft_cross_entropy(outputs, targets, *args, **kwargs)
            return losses

        config['model']._loss_fn = dict_loss

        trainer = Trainer(**config)
        trainer.fit()
