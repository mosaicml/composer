# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import functools
import torch
from torchvision.models.resnet import Bottleneck


def stochastic_bottleneck_forward(module: torch.nn.Module, x: torch.Tensor, probability: float) -> torch.Tensor:
    identity = x

    sample = module.training or bool(torch.bernoulli(probability))

    if sample:
        out = module.conv1(x)
        out = module.bn1(out)
        out = module.relu(out)

        out = module.conv2(out)
        out = module.bn2(out)
        out = module.relu(out)

        out = module.conv3(out)
        out = module.bn3(out)

        if not module.training:
            out = out * (1 - probability)

        if module.downsample is not None:
            identity = module.downsample(x)

        out += identity
        out = module.relu(out)
    else:
        if module.downsample is not None:
            out = module.relu(module.downsample)
        else:
            out = identity
    return out


def make_resnet_bottleneck_stochastic(module: Bottleneck, module_index: int, module_count: int, drop_rate: float,
                                      drop_distribution: str):
    if drop_distribution == 'linear':
        drop_rate = ((module_index + 1) / module_count) * drop_rate

    module.forward = functools.partial(stochastic_bottleneck_forward, module=module, probability=drop_rate)
    return module