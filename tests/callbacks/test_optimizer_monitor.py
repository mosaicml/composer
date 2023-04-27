# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version
from torch.utils.data import DataLoader

from composer.callbacks import OptimizerMonitor
from composer.loggers import InMemoryLogger
from composer.models import HuggingFaceModel
from composer.optim import DecoupledAdamW
from composer.trainer import Trainer
from composer.utils import dist, using_torch_2
from tests.common import device, world_size
from tests.common.datasets import RandomClassificationDataset, RandomTextLMDataset
from tests.common.models import SimpleModel


@pytest.mark.parametrize('log_optimizer_metrics', [True, False])
@pytest.mark.parametrize('batch_log_interval', [1, 2, 3, 5])
def test_optimizer_monitor(log_optimizer_metrics: bool, batch_log_interval: int):
    # Construct the callback
    grad_monitor = OptimizerMonitor(log_optimizer_metrics=log_optimizer_metrics, batch_log_interval=batch_log_interval)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel()
    # Construct the trainer and train
    trainer = Trainer(
        model=model,
        callbacks=grad_monitor,
        loggers=in_memory_logger,
        train_dataloader=DataLoader(RandomClassificationDataset()),
        optimizers=DecoupledAdamW(model.parameters()),
        max_duration='10ba',
    )
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)
    expected_num_calls = num_train_steps // batch_log_interval

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['l2_norm/grad/global'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'l2_norm/grad' in k]
    assert 'l2_norm/grad/module.2.weight' in in_memory_logger.data.keys()
    if log_optimizer_metrics:
        assert 'l2_norm/moment/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/moment_grad/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/second_moment_sqrt/module.2.weight' in in_memory_logger.data.keys()
        assert 'l2_norm/update/module.2.weight' in in_memory_logger.data.keys()
        assert 'cosine/update_grad/module.2.weight' in in_memory_logger.data.keys()

    # Expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == expected_num_calls
    for num_calls in layer_norm_calls:
        assert num_calls == expected_num_calls


@device('gpu')
@world_size(1, 2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.parametrize('use_orig_params', [True, False])
def test_fsdp_optimizer_monitor(device, world_size, use_orig_params):
    if use_orig_params and not using_torch_2():
        pytest.skip('use_orig_params was introduced in pytorch 2.0')

    # Construct the callback
    grad_monitor = OptimizerMonitor(log_optimizer_metrics=True)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = SimpleModel(num_classes=100, num_features=100, num_hidden=100)
    for module in model.modules():
        if len(list(module.parameters())) > 0:
            module._fsdp_wrap = True
    dataset = RandomClassificationDataset(num_classes=100, shape=(100, 1, 1))
    # Construct the trainer and train
    trainer = Trainer(model=model,
                      callbacks=grad_monitor,
                      loggers=in_memory_logger,
                      train_dataloader=DataLoader(dataset, sampler=dist.get_sampler(dataset)),
                      optimizers=DecoupledAdamW(model.parameters()),
                      max_duration='3ba',
                      fsdp_config={
                          'sharding_strategy': 'FULL_SHARD' if world_size > 1 else 'NO_SHARD',
                          'cpu_offload': False,
                          'mixed_precision': 'PURE',
                          'backward_prefetch': 'BACKWARD_PRE',
                          'activation_checkpointing': False,
                          'activation_cpu_offload': False,
                          'verbose': False,
                          'use_orig_params': use_orig_params,
                      })
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['l2_norm/grad/global'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'l2_norm/grad' in k]
    suffix = ('._flat_param' if using_torch_2() else '.flat_param') if not use_orig_params else '.weight'
    infix = '' if using_torch_2() else '._fpw_module'
    test_keys = [
        f'l2_norm/grad/module._fsdp_wrapped_module{infix}.4._fsdp_wrapped_module',
        f'l2_norm/moment/module._fsdp_wrapped_module{infix}.4._fsdp_wrapped_module',
        f'cosine/moment_grad/module._fsdp_wrapped_module{infix}.4._fsdp_wrapped_module',
        f'l2_norm/second_moment_sqrt/module._fsdp_wrapped_module{infix}.4._fsdp_wrapped_module',
        f'l2_norm/update/module._fsdp_wrapped_module{infix}.4._fsdp_wrapped_module',
        f'cosine/update_grad/module._fsdp_wrapped_module{infix}.4._fsdp_wrapped_module',
    ]
    test_keys = [key + suffix for key in test_keys]
    for key in test_keys:
        assert key in in_memory_logger.data.keys()

    # Expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    for num_calls in layer_norm_calls:
        assert num_calls == num_train_steps


@device('gpu')
@world_size(1, 2)
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse('1.13.0'),
                    reason='requires PyTorch 1.13 or higher')
@pytest.mark.parametrize('use_orig_params', [True, False])
def test_fsdp_optimizer_monitor_transformer(device, world_size, tiny_gpt2_model, tiny_gpt2_tokenizer, use_orig_params):
    if use_orig_params and not using_torch_2():
        pytest.skip('use_orig_params was introduced in pytorch 2.0')
    transformers = pytest.importorskip('transformers')
    # Construct the callback
    grad_monitor = OptimizerMonitor(log_optimizer_metrics=True)
    in_memory_logger = InMemoryLogger()  # track the logged metrics in the in_memory_logger
    model = HuggingFaceModel(model=tiny_gpt2_model, tokenizer=tiny_gpt2_tokenizer, use_logits=True)
    model.model.lm_head._fsdp_wrap = False
    model.model.transformer._fsdp_wrap = False
    for block in model.model.transformer.h:
        block._fsdp_wrap = True
    train_dataset = RandomTextLMDataset(size=8,
                                        vocab_size=tiny_gpt2_model.config.vocab_size,
                                        sequence_length=4,
                                        use_keys=True,
                                        use_token_type_ids=False,
                                        conditional_generation=False,
                                        causal_lm=True)

    collator = transformers.DefaultDataCollator()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  collate_fn=collator,
                                  sampler=dist.get_sampler(train_dataset))
    # Construct the trainer and train
    trainer = Trainer(model=model,
                      callbacks=grad_monitor,
                      loggers=in_memory_logger,
                      train_dataloader=train_dataloader,
                      optimizers=DecoupledAdamW(model.parameters()),
                      max_duration='3ba',
                      fsdp_config={
                          'sharding_strategy': 'FULL_SHARD' if world_size > 1 else 'NO_SHARD',
                          'cpu_offload': False,
                          'mixed_precision': 'PURE',
                          'backward_prefetch': 'BACKWARD_PRE',
                          'activation_checkpointing': False,
                          'activation_cpu_offload': False,
                          'verbose': False,
                          'use_orig_params': use_orig_params,
                      })
    trainer.fit()
    num_train_steps = int(trainer.state.timestamp.batch)

    # Count the logged steps
    grad_norm_calls = len(in_memory_logger.data['l2_norm/grad/global'])
    layer_norm_calls = [len(calls) for (k, calls) in in_memory_logger.data.items() if 'l2_norm/grad' in k]
    # an incomplete list of expected keys
    if not use_orig_params:
        suffix = '._flat_param' if using_torch_2() else '.flat_param'
        infix = '' if using_torch_2() else '._fpw_module'
        test_keys = [
            f'l2_norm/grad/model._fsdp_wrapped_module{infix}.transformer.h.1._fsdp_wrapped_module{suffix}',
            f'l2_norm/update/model._fsdp_wrapped_module{infix}.transformer.h.1._fsdp_wrapped_module{suffix}',
            f'cosine/moment_grad/model._fsdp_wrapped_module{infix}.transformer.h.1._fsdp_wrapped_module{suffix}',
            f'cosine/update_grad/model._fsdp_wrapped_module{infix}.transformer.h.1._fsdp_wrapped_module{suffix}',
            f'cosine/moment_grad/model._fsdp_wrapped_module{infix}.transformer.h.1._fsdp_wrapped_module{suffix}',
        ]
    else:
        test_keys = [
            'l2_norm/grad/model._fsdp_wrapped_module.transformer.h.1._fsdp_wrapped_module.mlp.c_proj.weight',
            'l2_norm/update/model._fsdp_wrapped_module.transformer.h.1._fsdp_wrapped_module.mlp.c_proj.weight',
            'cosine/moment_grad/model._fsdp_wrapped_module.transformer.h.1._fsdp_wrapped_module.attn.c_attn.weight',
            'cosine/update_grad/model._fsdp_wrapped_module.transformer.h.1._fsdp_wrapped_module.attn.c_attn.weight',
            'cosine/moment_grad/model._fsdp_wrapped_module.transformer.h.1._fsdp_wrapped_module.mlp.c_proj.weight',
        ]
    for key in test_keys:
        assert key in in_memory_logger.data.keys()

    # Expected to log gradient norm once per step (total batch)
    assert grad_norm_calls == num_train_steps
    for num_calls in layer_norm_calls:
        assert num_calls == num_train_steps
