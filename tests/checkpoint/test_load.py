# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import uuid
from pathlib import Path

import pytest
from torch.utils.data import DataLoader

from composer.checkpoint.load import (
    load_checkpoint,
    load_model_checkpoint,
    load_optim_checkpoint,
    load_resumption_checkpoint,
)
from composer.checkpoint.save import (
    save_checkpoint_to_disk,
    save_model_to_disk,
    save_optim_to_disk,
    save_resumption_state_to_disk,
)
from composer.checkpoint.state_dict import (
    _is_model_fsdp,
    get_model_state_dict,
    get_optim_state_dict,
    get_resumption_state_dict,
)
from composer.trainer import Trainer
from composer.utils import dist
from tests.checkpoint.helpers import init_model, init_model_and_optimizer, init_state
from tests.common import (
    RandomClassificationDataset,
)
from tests.common.compare import deep_compare


@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_model,sharded_checkpoint,shard_as_needed_during_load',
    [
        #Loading an unsharded checkpoint into an unsharded model on a single GPU (not sharding after)
        pytest.param(1, False, False, False, marks=pytest.mark.world_size(1)),

        # Loading a sharded checkpoint into a sharded model in distributed setting
        pytest.param(2, True, True, False, marks=pytest.mark.world_size(2)),

        # SHOULD FAIL: Loading an unsharded checkpoint into a sharded model
        pytest.param(2, True, False, False, marks=pytest.mark.world_size(2)),

        # SHOULD FAIL: Attempting to load a sharded checkpoint into an unsharded model without sharding
        pytest.param(2, False, True, False, marks=pytest.mark.world_size(2)),

        # Loading a sharded checkpoint into an unsharded model (sharding it before load)
        pytest.param(2, False, True, True, marks=pytest.mark.world_size(2)),

        # Loading an unsharded checkpoint into an unsharded model and sharding it after.
        pytest.param(2, False, False, True, marks=pytest.mark.world_size(2)),

        # The other three permutations of the above tests are:
        # 2 gpu, Sharded model, sharded checkpoint, with additional sharding -> no need to shard already sharded model
        # 2 gpu, Sharded model, unsharded checkpoint, with additional sharding -> no need to shard already sharded model
        # 2 gpu, Unsharded model, unsharded checkpoint, without additional sharding -> no need to try this on 2 gpus
    ],
)
def test_load_model_checkpoint(
    world_size: int,
    tmp_path: Path,
    sharded_model: bool,
    sharded_checkpoint: bool,
    shard_as_needed_during_load: bool,
):
    if sharded_model and not sharded_checkpoint:
        pytest.xfail(
            'Loading an unsharded checkpoint into a sharded model is not supported and causes OOMs when running with these tests',
        )
    # Ensure all ranks use the same path
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]

    # Save a model checkpoint
    model, _ = init_model(use_fsdp=sharded_checkpoint, device='cuda')
    save_path = os.path.join(destination_dir, 'model.pt') if not sharded_checkpoint else destination_dir
    saved_path = save_model_to_disk(model, save_path, sharded_checkpoint=sharded_checkpoint)

    # Get the original model's state dict
    original_state_dict = get_model_state_dict(model, sharded_state_dict=False)
    # Load the model checkpoint
    new_model, _ = init_model(use_fsdp=sharded_model, device='cuda')
    if saved_path is not None:
        load_path = saved_path if not sharded_checkpoint else str(Path(saved_path).parent)
    else:
        load_path = ''

    if not sharded_model and sharded_checkpoint and not shard_as_needed_during_load:
        context_manager = pytest.raises(ValueError)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        load_model_checkpoint(
            new_model,
            load_path=load_path,
            load_options=dict(
                sharded_checkpoint=sharded_checkpoint,
                shard_as_needed_during_load=shard_as_needed_during_load,
            ),
        )
        # Check if model is sharded when it should be
        if shard_as_needed_during_load:
            assert _is_model_fsdp(new_model), 'Model should be sharded after load'

        # Get the new model's state dict
        new_state_dict = get_model_state_dict(new_model, sharded_state_dict=False)

        if dist.get_global_rank() == 0:
            deep_compare(original_state_dict, new_state_dict)


@pytest.mark.filterwarnings('ignore:TypedStorage is deprecated.')
@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_optimizer,sharded_checkpoint,shard_as_needed_during_load',
    [
        # Loading an unsharded checkpoint into an unsharded optimizer on a single GPU (not sharding after)
        pytest.param(1, False, False, False, marks=pytest.mark.world_size(1)),

        # Loading a sharded checkpoint into a sharded optimizer in distributed setting
        pytest.param(2, True, True, False, marks=pytest.mark.world_size(2)),
    ],
)
def test_load_optim_checkpoint(
    world_size: int,
    tmp_path: Path,
    sharded_optimizer: bool,
    sharded_checkpoint: bool,
    shard_as_needed_during_load: bool,
):
    if sharded_optimizer and not sharded_checkpoint:
        pytest.xfail(
            'Loading an unsharded checkpoint into a sharded optimizer is not supported and causes OOMs when running with these tests',
        )

    # Ensure all ranks use the same path
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]

    # Save an optimizer checkpoint
    model, optimizer = init_model_and_optimizer(use_fsdp=sharded_checkpoint, device='cuda')
    save_path = os.path.join(destination_dir, 'optim.pt') if not sharded_checkpoint else destination_dir
    saved_path = save_optim_to_disk(model, optimizer, save_path, sharded_checkpoint=sharded_checkpoint)

    # Get the original optimizer's state dict
    original_state_dict = get_optim_state_dict(model, optimizer, sharded_state_dict=False)

    # Load the optimizer checkpoint
    new_model, new_optimizer = init_model_and_optimizer(use_fsdp=sharded_optimizer, device='cuda')
    if saved_path is not None:
        load_path = saved_path if not sharded_checkpoint else str(Path(saved_path).parent)
    else:
        load_path = ''

    if not sharded_optimizer and sharded_checkpoint and not shard_as_needed_during_load:
        context_manager = pytest.raises(ValueError)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        load_optim_checkpoint(
            new_model,
            new_optimizer,
            load_path=load_path,
            load_options=dict(
                sharded_checkpoint=sharded_checkpoint,
                shard_as_needed_during_load=shard_as_needed_during_load,
            ),
        )

        # Check if optimizer is sharded when it should be
        if shard_as_needed_during_load:
            assert _is_model_fsdp(new_model), 'Optimizer should be sharded after load'

        # Get the new optimizer's state dict
        new_state_dict = get_optim_state_dict(new_model, new_optimizer, sharded_state_dict=False)

        if dist.get_global_rank() == 0:
            deep_compare(original_state_dict, new_state_dict)


@pytest.mark.gpu
@pytest.mark.filterwarnings('ignore:SWA has')
def test_load_resumption_checkpoint(tmp_path: Path):
    # Ensure all ranks use the same path
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]

    # Create an initial state using the helper function
    initial_state = init_state(
        device='cpu',  # or 'cuda' if you want to test on GPU
        include_schedulers=True,
        include_algorithms=True,
        include_callbacks=True,
        use_grad_scaler=True,
        rank_zero_seed=42,
        run_name='test_run',
    )

    # Modify some values to ensure they're changed
    initial_state.timestamp.to_next_batch()
    initial_state.dataset_state = {'train': {'some_key': 'some_value'}}

    # Save the resumption state
    save_path = os.path.join(destination_dir, 'resumption.pkl')
    save_resumption_state_to_disk(initial_state, save_path)

    # Create a new state
    new_state = init_state(
        device='cpu',  # or 'cuda' if you want to test on GPU
        include_schedulers=True,
        include_algorithms=True,
        include_callbacks=True,
        use_grad_scaler=True,
        rank_zero_seed=42,
        run_name='test_run',
    )
    import time
    time.sleep(1)
    # Load the resumption checkpoint
    load_resumption_checkpoint(new_state, save_path)

    # Check that the loaded state matches the initial state
    deep_compare(new_state.timestamp.state_dict(), initial_state.timestamp.state_dict())
    assert new_state.rank_zero_seed == initial_state.rank_zero_seed
    assert new_state.run_name == initial_state.run_name

    # Check schedulers, algorithms, callbacks, and scaler
    if initial_state.schedulers:
        for init_scheduler, new_scheduler in zip(initial_state.schedulers, new_state.schedulers):
            deep_compare(init_scheduler.state_dict(), new_scheduler.state_dict())

    if initial_state.algorithms:
        for init_algo, new_algo in zip(initial_state.algorithms, new_state.algorithms):
            init_state_dict = init_algo.state_dict()
            new_state_dict = new_algo.state_dict()
            if 'repr' in init_state_dict:
                init_state_dict.pop('repr')
            if 'repr' in new_state_dict:
                new_state_dict.pop('repr')
            deep_compare(init_state_dict, new_state_dict)

    if initial_state.callbacks:
        for init_callback, new_callback in zip(initial_state.callbacks, new_state.callbacks):
            deep_compare(init_callback.state_dict(), new_callback.state_dict())

    if initial_state.scaler:
        assert initial_state.scaler is not None
        assert new_state.scaler is not None
        deep_compare(initial_state.scaler.state_dict(), new_state.scaler.state_dict())


@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_model,sharded_checkpoint,shard_as_needed_during_load',
    [
        # Loading an unsharded checkpoint into an unsharded model on a single GPU (not sharding after)
        pytest.param(1, False, False, False, marks=pytest.mark.world_size(1)),

        # Loading a sharded checkpoint into a sharded model in distributed setting
        pytest.param(2, True, True, False, marks=pytest.mark.world_size(2)),

        # SHOULD FAIL: Loading an unsharded checkpoint into a sharded model
        pytest.param(2, True, False, False, marks=pytest.mark.world_size(2)),

        # SHOULD FAIL: Attempting to load a sharded checkpoint into an unsharded model without sharding
        pytest.param(2, False, True, False, marks=pytest.mark.world_size(2)),
    ],
)
def test_load_checkpoint(
    world_size: int,
    tmp_path: Path,
    sharded_model: bool,
    sharded_checkpoint: bool,
    shard_as_needed_during_load: bool,
):

    if sharded_model and not sharded_checkpoint:
        pytest.xfail(
            'Loading an unsharded checkpoint into a sharded model is not supported and causes OOMs when running with these tests',
        )
    # Ensure all ranks use the same path
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]

    # Save an optimizer checkpoint
    state = init_state(
        use_fsdp=sharded_checkpoint,
        device='cuda',
        take_step=True,
    )
    load_path = save_checkpoint_to_disk(
        destination_dir=destination_dir,
        state=state,
        options={
            'sharded_checkpoint': sharded_checkpoint,
            'save_model': True,
            'save_optimizer': True,
            'save_resumption_state': True,
        },
    )
    original_model_state_dict = get_model_state_dict(state.model, sharded_state_dict=False)
    original_optim_state_dict = get_optim_state_dict(state.model, state.optimizers[0], sharded_state_dict=False)
    original_resumption_state = get_resumption_state_dict(state)
    new_state = init_state(use_fsdp=sharded_model, device='cuda', take_step=True)
    if not sharded_model and sharded_checkpoint and not shard_as_needed_during_load:
        context_manager = pytest.raises(ValueError)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        load_checkpoint(
            load_path=load_path,
            state=new_state,
            load_options={
                'sharded_checkpoint': sharded_checkpoint,
                'load_optimizer': True,
                'load_resumption_state': True,
                'shard_as_needed_during_load': shard_as_needed_during_load,
            },
        )
        if shard_as_needed_during_load:
            assert _is_model_fsdp(new_state.model), 'Model should be sharded after load'
        # Get the new model's state dict
        new_state_dict = get_model_state_dict(new_state.model, sharded_state_dict=False)
        new_optim_state_dict = get_optim_state_dict(new_state.model, new_state.optimizers[0], sharded_state_dict=False)
        new_resumption_state = get_resumption_state_dict(new_state)

        if dist.get_global_rank() == 0:
            deep_compare(original_model_state_dict, new_state_dict)
            deep_compare(original_optim_state_dict, new_optim_state_dict)
            deep_compare(original_resumption_state, new_resumption_state, ignore_keys=['rng', 'run_name'])


@pytest.mark.gpu
@pytest.mark.parametrize(
    'world_size,sharded_model,sharded_checkpoint,shard_as_needed_during_load',
    [
        # Loading an unsharded checkpoint into an unsharded model on a single GPU (not sharding after)
        pytest.param(1, False, False, False, marks=pytest.mark.world_size(1)),

        # Loading a sharded checkpoint into a sharded model in distributed setting
        pytest.param(2, True, True, False, marks=pytest.mark.world_size(2)),

        # Loading a sharded checkpoint into an unsharded model (sharding it before load)
        pytest.param(2, False, True, True, marks=pytest.mark.world_size(2)),

        # Loading an unsharded checkpoint into an unsharded model and sharding it after.
        pytest.param(2, False, False, True, marks=pytest.mark.world_size(2)),

        # The other three permutations of the above tests are:
        # 2 gpu, Sharded model, sharded checkpoint, with additional sharding -> no need to shard already sharded model
        # 2 gpu, Sharded model, unsharded checkpoint, with additional sharding -> no need to shard already sharded model
        # 2 gpu, Unsharded model, unsharded checkpoint, without additional sharding -> no need to try this on 2 gpus
    ],
)
def test_load_model_checkpoint_and_eval(
    world_size: int,
    tmp_path: Path,
    sharded_model: bool,
    sharded_checkpoint: bool,
    shard_as_needed_during_load: bool,
):
    if sharded_model and not sharded_checkpoint:
        pytest.xfail(
            'Loading an unsharded checkpoint into a sharded model is not supported and causes OOMs when running with these tests',
        )
    # Ensure all ranks use the same path
    destination_dir = os.path.join(tmp_path, str(uuid.uuid4())[:8])
    destination_dir = dist.all_gather_object(destination_dir)[0]

    # Save a model checkpoint
    model, _ = init_model(use_composer_model=True, use_fsdp=sharded_checkpoint, device='cuda')
    save_path = os.path.join(destination_dir, 'model.pt') if not sharded_checkpoint else destination_dir
    saved_path = save_model_to_disk(model, save_path, sharded_checkpoint=sharded_checkpoint)

    # Get the original model's state dict
    original_state_dict = get_model_state_dict(model, sharded_state_dict=False)
    # Load the model checkpoint
    new_model, _ = init_model(use_composer_model=True, use_fsdp=sharded_model, device='cuda')
    if saved_path is not None:
        load_path = saved_path if not sharded_checkpoint else str(Path(saved_path).parent)
    else:
        load_path = ''

    if not sharded_model and sharded_checkpoint and not shard_as_needed_during_load:
        context_manager = pytest.raises(ValueError)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:
        load_model_checkpoint(
            new_model,
            load_path=load_path,
            load_options=dict(
                sharded_checkpoint=sharded_checkpoint,
                shard_as_needed_during_load=shard_as_needed_during_load,
            ),
        )
        # Check if model is sharded when it should be
        if shard_as_needed_during_load:
            assert _is_model_fsdp(new_model), 'Model should be sharded after load'

        # Get the new model's state dict
        new_state_dict = get_model_state_dict(new_model, sharded_state_dict=False)

        if dist.get_global_rank() == 0:
            deep_compare(original_state_dict, new_state_dict)

        dataset = RandomClassificationDataset(
            shape=(8,),
            size=100,
            num_classes=3,
        )

        trainer = Trainer(
            eval_dataloader=DataLoader(
                dataset=dataset,
                sampler=dist.get_sampler(dataset),
            ),
            model=new_model,  # type: ignore
        )

        # Evaluate the model
        trainer.eval()
