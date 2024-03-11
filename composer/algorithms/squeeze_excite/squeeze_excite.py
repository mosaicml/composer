# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch
from torch.optim import Optimizer

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def apply_squeeze_excite(
    model: torch.nn.Module,
    latent_channels: float = 64,
    min_channels: int = 128,
    optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None,
) -> None:
    """Adds Squeeze-and-Excitation blocks (`Hu et al, 2019 <https://arxiv.org/abs/1709.01507>`_) after
    :class:`torch.nn.Conv2d` layers.

    A Squeeze-and-Excitation block applies global average pooling to the input,
    feeds the resulting vector to a single-hidden-layer fully-connected
    network (MLP), and uses the outputs of this MLP as attention coefficients
    to rescale the input. This allows the network to take into account global
    information about each input, as opposed to only local receptive fields
    like in a convolutional layer.

    Args:
        model (torch.nn.Module): The module to apply squeeze excite replacement to.
        latent_channels (float, optional): Dimensionality of the hidden layer within the added
            MLP. If less than 1, interpreted as a fraction of the number of
            output channels in the :class:`torch.nn.Conv2d` immediately
            preceding each Squeeze-and-Excitation block. Default: ``64``.
        min_channels (int, optional): An SE block is added after a :class:`torch.nn.Conv2d`
            module ``conv`` only if one of the layer's input or output channels is greater than
            this threshold. Default: ``128``.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional):
            Existing optimizer(s) bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so that
            they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_stochastic_depth(
                model,
                target_layer_name='ResNetBottleneck'
            )
    """

    def convert_module(module: torch.nn.Module, module_index: int):
        assert isinstance(module, torch.nn.Conv2d), 'should only be called with conv2d'
        already_squeeze_excited = hasattr(module, '_already_squeeze_excited') and module._already_squeeze_excited
        if min(module.in_channels, module.out_channels) >= min_channels and not already_squeeze_excited:
            return SqueezeExciteConv2d.from_conv2d(module, module_index, latent_channels=latent_channels)

    module_surgery.replace_module_classes(model, optimizers=optimizers, policies={torch.nn.Conv2d: convert_module})


class SqueezeExcite2d(torch.nn.Module):
    """Squeeze-and-Excitation block from (`Hu et al, 2019 <https://arxiv.org/abs/1709.01507>`_)

    This block applies global average pooling to the input, feeds the resulting
    vector to a single-hidden-layer fully-connected network (MLP), and uses the
    outputs of this MLP as attention coefficients to rescale the input. This
    allows the network to take into account global information about each input,
    as opposed to only local receptive fields like in a convolutional layer.

    Args:
        num_features (int): Number of features or channels in the input.
        latent_channels (float, optional): Dimensionality of the hidden layer within the added
            MLP. If less than 1, interpreted as a fraction of ``num_features``. Default: ``0.125``.
    """

    def __init__(self, num_features: int, latent_channels: float = .125):
        super().__init__()
        self.latent_channels = int(latent_channels if latent_channels >= 1 else latent_channels * num_features)
        flattened_dims = num_features

        self.pool_and_mlp = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(flattened_dims, self.latent_channels, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(self.latent_channels, num_features, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        n, c, _, _ = input.shape
        attention_coeffs = self.pool_and_mlp(input)
        return input * attention_coeffs.reshape(n, c, 1, 1)


class SqueezeExciteConv2d(torch.nn.Module):
    """Helper class used to add a :class:`.SqueezeExcite2d` module after a :class:`torch.nn.Conv2d`."""

    def __init__(self, *args, latent_channels: float = 0.125, conv: Optional[torch.nn.Conv2d] = None, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(*args, **kwargs) if conv is None else conv
        self.conv._already_squeeze_excited = True  # Mark to avoid rewrapping on duplicate calls # pyright: ignore[reportGeneralTypeIssues]
        self.se = SqueezeExcite2d(num_features=self.conv.out_channels, latent_channels=latent_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(input))

    @staticmethod
    def from_conv2d(module: torch.nn.Conv2d, module_index: int, latent_channels: float):
        return SqueezeExciteConv2d(conv=module, latent_channels=latent_channels)


class SqueezeExcite(Algorithm):
    """Adds Squeeze-and-Excitation blocks (`Hu et al, 2019 <https://arxiv.org/abs/1709.01507>`_) after the
    :class:`torch.nn.Conv2d` modules in a neural network.

    Runs on :attr:`.Event.INIT`. See :class:`SqueezeExcite2d` for more information.

    Args:
        latent_channels (float, optional): Dimensionality of the hidden layer within the added
            MLP. If less than 1, interpreted as a fraction of the number of
            output channels in the :class:`torch.nn.Conv2d` immediately
            preceding each Squeeze-and-Excitation block. Default: ``64``.
        min_channels (int, optional): An SE block is added after a :class:`torch.nn.Conv2d`
            module ``conv`` only if ``min(conv.in_channels, conv.out_channels) >= min_channels``.
            For models that reduce spatial size and increase channel count
            deeper in the network, this parameter can be used to only
            add SE blocks deeper in the network. This may be desirable
            because SE blocks add less overhead when their inputs have
            smaller spatial size. Default: ``128``.
    """

    def __init__(
        self,
        latent_channels: float = 64,
        min_channels: int = 128,
    ):
        self.latent_channels = latent_channels
        self.min_channels = min_channels

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(latent_channels={self.latent_channels},min_channels={self.min_channels})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        apply_squeeze_excite(
            state.model,
            optimizers=state.optimizers,
            latent_channels=self.latent_channels,
            min_channels=self.min_channels,
        )
        layer_count = module_surgery.count_module_instances(state.model, SqueezeExciteConv2d)

        log.info(
            f'Applied SqueezeExcite to model {state.model.__class__.__name__} '
            f'with latent_channels={self.latent_channels}, '
            f'min_channels={self.min_channels}. '
            f'Model now has {layer_count} SqueezeExcite layers.',
        )

        logger.log_hyperparameters({
            'squeeze_excite/num_squeeze_excite_layers': layer_count,
        })
