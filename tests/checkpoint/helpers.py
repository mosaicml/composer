# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union
from unittest.mock import MagicMock

import torch
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload
from torch.optim import adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from composer.algorithms import SWA
from composer.callbacks import SpeedMonitor
from composer.core import State
from composer.devices import Device, DeviceCPU, DeviceGPU
from composer.distributed import prepare_fsdp_module
from composer.models import ComposerModel
from composer.utils.parallelism import FSDPConfig
from tests.common.models import EvenSimplerMLP, SimpleComposerMLP

__all__ = [
    'init_model_and_optimizer',
    'init_model',
    'init_optimizer',
    'init_state',
]


def init_state(
    use_fsdp: bool = False,
    device: str = 'cpu',
    include_schedulers=False,
    include_callbacks=False,
    include_algorithms=False,
    use_grad_scaler=False,
    rank_zero_seed=10,
    run_name='test_run',
    take_step=False,
    wrap_with_raw_fsdp=False,
) -> State:
    model, optimizer = init_model_and_optimizer(
        use_fsdp=use_fsdp,
        use_composer_model=True,
        take_step=take_step,
        device=device,
        wrap_with_raw_fsdp=wrap_with_raw_fsdp,
    )

    test_dataset_sd = {'test': 0}
    device_obj: Device = DeviceCPU() if device == 'cpu' else DeviceGPU()

    dataloader = MagicMock(spec=DataLoader)
    dataloader.dataset = MagicMock()
    dataloader.dataset.state_dict = MagicMock(return_value=test_dataset_sd)
    kwargs = {}

    if include_callbacks:
        kwargs['callbacks'] = [SpeedMonitor(), SpeedMonitor()]
    if include_algorithms:
        kwargs['algorithms'] = [SWA()]
    if use_grad_scaler:
        if version.parse(torch.__version__) >= version.parse('2.3.0'):
            from torch.amp.grad_scaler import GradScaler
        else:
            from torch.cuda.amp.grad_scaler import GradScaler
        kwargs['scaler'] = GradScaler()

    state = State(
        model=model,
        rank_zero_seed=rank_zero_seed,
        run_name=run_name,
        device=device_obj,
        train_dataloader=dataloader,
        optimizers=[optimizer],
        **kwargs,
    )
    if include_schedulers:
        state.schedulers = StepLR(optimizer=optimizer, step_size=2)
    return state


def init_model_and_optimizer(
    use_composer_model: bool = True,
    num_classes=3,
    batch_size=5,
    num_features=8,
    take_step=True,
    use_fsdp=False,
    tensor_type='sharded_tensor',
    device='cuda',
    wrap_with_raw_fsdp=False,
) -> tuple[Union[ComposerModel, torch.nn.Module], torch.optim.Optimizer]:
    model, loss_fn = init_model(
        use_composer_model,
        num_classes=num_classes,
        num_features=num_features,
        use_fsdp=use_fsdp,
        tensor_type=tensor_type,
        device=device,
        wrap_with_raw_fsdp=wrap_with_raw_fsdp,
    )

    optimizer = init_optimizer(
        model,
        loss_fn,
        use_composer_model=use_composer_model,
        num_classes=num_classes,
        batch_size=batch_size,
        num_features=num_features,
        take_step=take_step,
        device=device,
    )

    return model, optimizer


def init_model(
    use_composer_model: bool = False,
    num_classes=3,
    num_features=8,
    use_fsdp=False,
    device='cuda',
    tensor_type='sharded_tensor',
    sync_module_states=True,
    cpu_offload=False,
    wrap_with_raw_fsdp=False,
) -> tuple[Union[ComposerModel, torch.nn.Module], Any]:
    if use_composer_model:
        model = SimpleComposerMLP(num_features=num_features, num_classes=num_classes, device=device)
        loss_fn = model._loss_fn
    else:
        model = EvenSimplerMLP(num_features=num_features, num_out_features=num_classes, device=device)
        loss_fn = torch.nn.CrossEntropyLoss()

    if use_fsdp:
        fsdp_kwargs: dict[str, Any] = dict(
            use_orig_params=True,
            sync_module_states=sync_module_states,  # To enable easy comparison between rank 0 unsharded model and full state dict
            cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
        )

        if tensor_type == 'dtensor':
            from torch.distributed.device_mesh import init_device_mesh
            device_mesh = init_device_mesh('cuda', (2,))
            fsdp_kwargs['device_mesh'] = device_mesh

        if wrap_with_raw_fsdp:
            model = FSDP(model, **fsdp_kwargs)
        else:
            if 'device_mesh' in fsdp_kwargs:
                mesh = fsdp_kwargs.pop('device_mesh')
                ndim = mesh.ndim
                if ndim == 1:
                    fsdp_kwargs['data_parallel_shard_degree'] = mesh.size(0)
                elif ndim == 2:
                    fsdp_kwargs['data_parallel_replicate_degree'] = mesh.size(0)
                    fsdp_kwargs['data_parallel_shard_degree'] = mesh.size(1)
                else:
                    raise ValueError(f'Unsupported device mesh dimension: {ndim}')

            prepare_fsdp_module(
                model,
                optimizers=None,
                fsdp_config=FSDPConfig(**fsdp_kwargs),
            )

    return model, loss_fn


def init_optimizer(
    model,
    loss_fn,
    use_composer_model: bool = False,
    num_classes=3,
    batch_size=5,
    num_features=8,
    take_step=True,
    device='cuda',
):
    inputs = torch.randn(batch_size, num_features, device=device)
    targets = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device, dtype=torch.long)
    batch = (inputs, targets) if use_composer_model else inputs
    optimizer = adam.Adam(model.parameters())
    outputs = model(batch)
    loss = loss_fn(outputs, targets)
    loss.backward()
    if take_step:
        optimizer.step()
    return optimizer
