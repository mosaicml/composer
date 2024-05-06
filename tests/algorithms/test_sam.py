# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.utils.data import DataLoader

from composer import Trainer
from composer.algorithms import SAM
from composer.algorithms.sam import SAMOptimizer
from composer.loss import soft_cross_entropy
from composer.optim import CosineAnnealingWithWarmupScheduler
from composer.utils import dist
from tests.common import RandomClassificationDataset, SimpleModel, world_size


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


@pytest.mark.gpu
@world_size(1, 2)
@pytest.mark.filterwarnings('ignore::UserWarning')
class TestSAMParamGroups():

    @pytest.fixture(params=['FSDP', 'DeepSpeed'])
    def config(self, request):

        distributed_mode = request.param

        train_dataset = RandomClassificationDataset(size=16)

        scheduler = CosineAnnealingWithWarmupScheduler(
            alpha_f=0.1,
            t_warmup='0.01dur',
        )

        config_dict = {
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
            'schedulers':
                scheduler,
            'precision':
                'amp_bf16',
            'fsdp_config':
                None,
            'deepspeed_config':
                None,
        }

        if distributed_mode == 'FSDP':
            config_dict['fsdp_config'] = {'sharding_strategy': 'NO_SHARD'}
        else:
            config_dict['deepspeed_config'] = {'prescale_gradients': True}

        # Simulate world_size checking as in LLMFoundry. See:
        # * https://github.com/mosaicml/llm-foundry/blob/bfbb8c57053eaa3cb99a5d51ba602d1a6c872aa7/scripts/train/train.py#L519-L523
        if dist.get_world_size(
        ) == 1 and (config_dict['fsdp_config'] is not None or config_dict['deepspeed_config'] is not None):
            config_dict['fsdp_config'] = config_dict['deepspeed_config'] = None

        return config_dict

    def test_param_groups_id_matching(self, config, world_size: int):
        trainer = Trainer(**config)

        sam_optimizer: SAMOptimizer = trainer.state.optimizers[0]
        base_optimizer: torch.optim.Optimizer = sam_optimizer.base_optimizer

        # Both SAMOptimizer and base_optimizer have to reference the same param groups
        assert id(sam_optimizer.param_groups[0]) == id(base_optimizer.param_groups[0])

    def test_params_value_close_after_updating(self, config, world_size: int):
        trainer = Trainer(**config)
        trainer.fit()

        sam_optimizer: SAMOptimizer = trainer.state.optimizers[0]
        base_optimizer: torch.optim.Optimizer = sam_optimizer.base_optimizer

        # If SAMOptimizer and base_optimizer reference the same param groups, then
        # the params are synchronized after updating (e.g. `lr` by a LR scheduler, 
        # weights by an optimizer step, etc.)
        assert math.isclose(
            sam_optimizer.param_groups[0]['lr'],
            base_optimizer.param_groups[0]['lr'],
        )