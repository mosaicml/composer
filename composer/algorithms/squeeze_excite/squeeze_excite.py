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
    """See :class:`SqueezeExcite`"""

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
    """Squeeze-and-Excitation block from (`Hu et al. 2019 <https://arxiv.org/abs/1709.01507>`_)

    This block applies global average pooling to the input, feeds the resulting
    vector to a single-hidden-layer fully-connected network (MLP), and uses the
    output of this MLP as attention coefficients to rescale the input. This
    allows the network to take into account global information about each input,
    as opposed to only local receptive fields like in a convolutional layer.

    Args:
        num_features: Number of features or channels in the input
        latent_channels: Dimensionality of the hidden layer within the added
            MLP. If less than 1, interpreted as a fraction of ``num_features``.
    """

    def __init__(self, num_features: int, latent_channels: float = .125):
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
    """Helper class used to add a :class:`SqueezeExcite2d` module after a :class:`~torch.nn.Conv2d`."""

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
    """See :class:`SqueezeExcite`"""

    def convert_module(module: torch.nn.Conv2d, module_index: int):
        if min(module.in_channels, module.out_channels) < min_channels:
            return None
        return SqueezeExciteConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)

    transforms = {torch.nn.Conv2d: convert_module}
    surgery.replace_module_classes(model, transforms)  # type: ignore
    return model


class SqueezeExcite(Algorithm):
    """Adds Squeeze-and-Excitation blocks (`Hu et al. 2019 <https://arxiv.org/abs/1709.01507>`_) after the :class:`~torch.nn.Conv2d` modules in a neural network.

    See :class:`SqueezeExcite2d` for more information.

    Args:
        latent_channels: Dimensionality of the hidden layer within the added
            MLP. If less than 1, interpreted as a fraction of ``num_features``.
        min_channels: An SE block is added after a :class:`~torch.nn.Conv2d`
            module ``conv`` only if
            ``min(conv.in_channels, conv.out_channels) >= min_channels``.
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
        self.hparams = SqueezeExciteHparams(
            latent_channels=latent_channels,
            min_channels=min_channels,
        )

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.INIT

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run no         
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Apply the Squeeze-and-Excitation layer replacement.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger        
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
