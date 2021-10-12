# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import yahp as hp

from composer.algorithms.algorithm_hparams import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)


@dataclass
class SqueezeExciteHparams(AlgorithmHparams):

    latent_channels: float = hp.optional(
        doc='Dimensionality of hidden layer within the added MLP.',
        default=64,
    )
    min_channels: int = hp.optional(
        doc='Minimum number of channels in a Conv2d layer'
        ' for a squeeze-excite block to be placed after it.',
        default=128,
    )

    def initialize_object(self) -> SqueezeExcite:
        return SqueezeExcite(**asdict(self))


class SqueezeExcite2d(torch.nn.Module):
    """`Squeeze-and-Excitation block <https://arxiv.org/abs/1709.01507>`_"""

    def __init__(self, num_features, latent_channels=.125):
        super().__init__()
        self.latent_channels = int(latent_channels if latent_channels >= 1 else latent_channels * num_features)
        flattened_dims = num_features

        self.pool_and_mlp = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                                torch.nn.Linear(flattened_dims, self.latent_channels, bias=False),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.latent_channels, num_features, bias=False),
                                                torch.nn.Sigmoid())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = input.shape
        attention_coeffs = self.pool_and_mlp(input)
        return input * attention_coeffs.reshape(n, c, 1, 1)


class SqueezeExciteConv2d(torch.nn.Module):
    """Helper class used to add a `Squeeze-and-Excitation block <https://arxiv.org/abs/1709.01507>`_ after a Conv2d."""

    def __init__(self, *args, latent_channels=.125, conv: torch.nn.Conv2d = None, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs) if conv is None else conv
        self.se = SqueezeExcite2d(num_features=self.conv.out_channels, latent_channels=latent_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(input))

    @staticmethod
    def from_conv2d(module: torch.nn.Conv2d, module_index: int, latent_channels: float):
        return SqueezeExciteConv2d(conv=module, latent_channels=latent_channels)


def apply_se(model: torch.nn.Module, latent_channels: float, min_channels: int):
    """Adds Squeeze-and-Excitation <https://arxiv.org/abs/1709.01507>`_  (SE) blocks
    after the `Conv2d` layers of a neural network.

    Args:
        model: A module containing one or more `torch.nn.Conv2d` modules.
        latent_channels: The dimensionality of the hidden layer within the added MLP.
        min_channels: An SE block is added after a `Conv2d` module `conv`
            only if `min(conv.in_channels, conv.out_channels) >= min_channels`.
            For models that reduce spatial size and increase channel count
            deeper in the network, this parameter can be used to only
            add SE blocks deeper in the network. This may be desirable
            because SE blocks add less overhead when their inputs have
            smaller spatial size.
    """

    def convert_module(module: torch.nn.Conv2d, module_index: int):
        if min(module.in_channels, module.out_channels) < min_channels:
            return None
        return SqueezeExciteConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)

    transforms = {torch.nn.Conv2d: convert_module}
    surgery.replace_module_classes(model, transforms)  # type: ignore
    return model


class SqueezeExcite(Algorithm):
    """Adds Squeeze-and-Excitation <https://arxiv.org/abs/1709.01507>`_  (SE) blocks
    after the `Conv2d` layers of a neural network.

    Args:
        latent_channels: The dimensionality of the hidden layer within the added MLP.
        min_channels: An SE block is added after a `Conv2d` module `conv`
            only if `min(conv.in_channels, conv.out_channels) >= min_channels`.
            For models that reduce spatial size and increase channel count
            deeper in the network, this parameter can be used to only
            add SE blocks deeper in the network. This may be desirable
            because SE blocks add less overhead when their inputs have
            smaller spatial size.
    """

    def __init__(
        self,
        latent_channels: float = 64,
        min_channels: int = 128,
    ):
        """
        __init__ is constructed from the same fields as in hparams.
        """
        self.hparams = SqueezeExciteHparams(
            latent_channels=latent_channels,
            min_channels=min_channels,
        )

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """
        Apply the Squeeze-and-Excitation layer replacement.
        """
        state.model = apply_se(state.model,
                               latent_channels=self.hparams.latent_channels,
                               min_channels=self.hparams.min_channels)
        layer_count = surgery.count_module_instances(state.model, SqueezeExciteConv2d)

        log.info(f'Applied SqueezeExcite to model {state.model.__class__.__name__} '
                 f'with latent_channels={self.hparams.latent_channels}, '
                 f'min_channels={self.hparams.min_channels}. '
                 f'Model now has {layer_count} SqueezeExcite layers.')

        logger.metric_fit({
            'squeeze_excite/num_squeeze_excite_layers': layer_count,
        })
