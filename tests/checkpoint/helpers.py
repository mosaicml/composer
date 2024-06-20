# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload
from torch.optim import adam

from tests.common.models import EvenSimplerMLP, SimpleComposerMLP

__all__ = [
    'init_model_and_optimizer',
    'init_model',
    'init_optimizer',
]


def init_model_and_optimizer(
    use_composer_model: bool,
    num_classes=3,
    batch_size=5,
    num_features=8,
    take_step=True,
    use_fsdp=False,
    tensor_type='sharded_tensor',
    device='cuda',
):
    model, loss_fn = init_model(
        use_composer_model,
        num_classes=num_classes,
        num_features=num_features,
        use_fsdp=use_fsdp,
        tensor_type=tensor_type,
        device=device,
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
):
    if use_composer_model:
        model = SimpleComposerMLP(num_features=num_features, num_classes=num_classes, device=device)
        loss_fn = model._loss_fn
    else:
        model = EvenSimplerMLP(num_features=num_features, num_out_features=num_classes, device=device)
        loss_fn = torch.nn.CrossEntropyLoss()

    if use_fsdp:
        fsdp_kwargs: Dict[str, Any] = dict(
            use_orig_params=True,
            sync_module_states=sync_module_states,  # To enable easy comparison between rank 0 unsharded model and full state dict
            cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
            device_id=torch.device('cpu') if device == 'cpu' else None,
        )

        if tensor_type == 'dtensor':
            from torch.distributed.device_mesh import init_device_mesh
            device_mesh = init_device_mesh('cuda', (2,))
            fsdp_kwargs['device_mesh'] = device_mesh

        model = FSDP(
            model,
            **fsdp_kwargs,
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
