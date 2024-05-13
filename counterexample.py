# Copyright 2024 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import torch
import torch.distributed as tdist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils.device import get_device


class CounterExampleModel(torch.nn.Module):
    """Small classification model.

    Args:
        num_features (int): number of input features (default: 1)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(
        self,
        num_features: int = 1,
        num_classes: int = 2,
        num_hidden: int = 8,
        device: str = 'cpu',
        bias: bool = True,
    ) -> None:

        self.num_features = num_features
        self.num_classes = num_classes

        fc1 = torch.nn.Linear(num_features, num_hidden, device=device, bias=bias)
        fc2 = torch.nn.Linear(num_hidden, num_classes, device=device, bias=bias)

        net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.Softmax(dim=-1),
        )
        net.param_init_fn = self.param_init_fn  # pyright: ignore[reportGeneralTypeIssues]
        super().__init__()

        self.net = net
        # Important: It is crucial that the FC layers are bound to `self`
        # for the optimizer surgery tests.
        # These tests attempt to perform surgery on `fc1` layer, and we want
        # to make sure that post-surgery, self.fc1 refers to the same parameters
        # as self.net[1]
        self.fc1 = fc1
        self.fc2 = fc2

    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=0.1)

        if isinstance(module, torch.nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                torch.nn.init.zeros_(module.bias)


if __name__ == '__main__':
    tdist.init_process_group(backend='gloo')

    torch_model = CounterExampleModel()

    fsdp_config = {
        'use_orig_params': True,
        'state_dict_type': 'full',
    }

    device = get_device('gpu')
    prepare_fsdp_module(
        model=torch_model,
        fsdp_config=fsdp_config,
        optimizers=None,
        precision='amp_bf16',
        device=device,
        auto_microbatching=True,
    )
    fsdp_model = torch_model

    torch_state_dict = get_model_state_dict(
        fsdp_model,
        submodules=None,
        options=StateDictOptions(full_state_dict=True),
    )

    if tdist.get_rank() == 0:
        print(f"{torch_state_dict['fc2.bias']=}")
        assert len(torch_state_dict['fc2.bias']) != 0
