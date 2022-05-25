# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import functools
import torch
from torchvision.models.resnet import Bottleneck


def block_stochastic_bottleneck_forward(x: torch.Tensor, module: torch.nn.Module,
                                        drop_rate: torch.Tensor) -> torch.Tensor:
    identity = x

    sample = (not module.training) or bool(torch.bernoulli(1 - drop_rate))
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
            out = out * (1 - drop_rate)

        if module.downsample is not None:
            identity = module.downsample(x)

        out += identity
        out = module.relu(out)
    else:
        if module.downsample is not None:
            out = module.relu(module.downsample(identity))
        else:
            out = identity
    return out


def _sample_drop(x: torch.Tensor, sample_drop_rate: float, is_training: bool):
    """Randomly drops samples from the input batch according to the `sample_drop_rate`.

    This is implemented by setting the samples to be dropped to zeros.
    """

    keep_probability = (1 - sample_drop_rate)
    if not is_training:
        return x * keep_probability
    rand_dim = [x.shape[0]] + [1] * len(x.shape[1:])
    sample_mask = keep_probability + torch.rand(rand_dim, dtype=x.dtype, device=x.device)
    sample_mask.floor_()  # binarize
    x *= sample_mask
    return x


def sample_stochastic_bottleneck_forward(x: torch.Tensor, module: torch.nn.Module, drop_rate: torch.Tensor):
    identity = x

    out = module.conv1(x)
    out = module.bn1(out)
    out = module.relu(out)

    out = module.conv2(out)
    out = module.bn2(out)
    out = module.relu(out)

    out = module.conv3(out)
    out = module.bn3(out)

    if module.downsample is not None:
        identity = module.downsample(x)

    if drop_rate:
        out = _sample_drop(out, drop_rate, module.training)
    out += identity

    return module.relu(out)


def make_resnet_bottleneck_stochastic(module: Bottleneck, module_index: int, module_count: int, drop_rate: float,
                                      drop_distribution: str, stochastic_method: str):
    if drop_distribution == 'linear':
        drop_rate = ((module_index + 1) / module_count) * drop_rate
    stochastic_func = block_stochastic_bottleneck_forward if stochastic_method == 'block' else sample_stochastic_bottleneck_forward
    module.forward = functools.partial(stochastic_func, module=module, drop_rate=torch.tensor(drop_rate))
    print(module.forward)
    return module