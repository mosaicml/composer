# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torchvision.models.resnet import Bottleneck


def block_stochastic_bottleneck_forward(module: torch.nn.Module, drop_rate: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, 'drop_rate'):
        module.drop_rate = drop_rate

    def forward(self, x):
        identity = x

        sample = (not self.training) or bool(torch.bernoulli(1 - self.drop_rate))

        if sample:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if not self.training:
                out = out * (1 - self.drop_rate)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)
        else:
            if self.downsample is not None:
                out = self.relu(self.downsample(identity))
            else:
                out = identity
        return out

    return forward.__get__(module)


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


def sample_stochastic_bottleneck_forward(module: torch.nn.Module, drop_rate: torch.Tensor):
    if not hasattr(module, 'drop_rate'):
        module.drop_rate = drop_rate

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.drop_rate:
            out = _sample_drop(out, self.drop_rate, self.training)
        out += identity

        return self.relu(out)

    return forward.__get__(module)


def make_resnet_bottleneck_stochastic(module: Bottleneck, module_index: int, module_count: int, drop_rate: float,
                                      drop_distribution: str, stochastic_method: str):
    if drop_distribution == 'linear':
        drop_rate = ((module_index + 1) / module_count) * drop_rate
    module.drop_rate = torch.tensor(drop_rate)

    stochastic_func = block_stochastic_bottleneck_forward if stochastic_method == 'block' else sample_stochastic_bottleneck_forward
    module.forward = stochastic_func(module=module, drop_rate=torch.tensor(drop_rate))

    return module
