# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameters for optimizers."""

from abc import ABC
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Type, Union

import torch
import torch_optimizer
import yahp as hp
from torch.optim import Optimizer

from composer.optim import DecoupledAdamW, DecoupledSGDW

# Optimizer parameters and defaults match those in torch.optim

__all__ = [
    'OptimizerHparams',
    'AdamHparams',
    'RAdamHparams',
    'AdamWHparams',
    'DecoupledAdamWHparams',
    'SGDHparams',
    'DecoupledSGDWHparams',
    'RMSpropHparams',
]


@dataclass
class OptimizerHparams(hp.Hparams, ABC):
    """Base class for optimizer hyperparameter classes.

    Optimizer parameters that are added to :class:`~composer.trainer.trainer_hparams.TrainerHparams` (e.g. via YAML or
    the CLI) are initialized in the training loop.
    """

    optimizer_cls = None  # type: Optional[Type[Optimizer]]

    def initialize_object(
        self,
        param_group: Union[Iterable[torch.Tensor], Iterable[Dict[str, torch.Tensor]]],
    ) -> Optimizer:
        """Initializes the optimizer.

        Args:
            param_group (Iterable[torch.Tensor] | Iterable[Dict[str, torch.Tensor]]): Parameters for
                this optimizer to optimize.
        """
        if self.optimizer_cls is None:
            raise ValueError(f'{type(self).__name__}.optimizer_cls must be defined')
        return self.optimizer_cls(param_group, **asdict(self))


@dataclass
class AdamHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.Adam` optimizer.

    See :class:`~torch.optim.Adam` for documentation.

    Args:
        lr (float, optional): See :class:`~torch.optim.Adam`.
        betas (float, optional): See :class:`~torch.optim.Adam`.
        eps (float, optional): See :class:`~torch.optim.Adam`.
        weight_decay (float, optional): See :class:`~torch.optim.Adam`.
        amsgrad (bool, optional): See :class:`~torch.optim.Adam`.
    """

    optimizer_cls = torch.optim.Adam

    lr: float = hp.auto(torch.optim.Adam, 'lr', ignore_docstring_errors=True)
    betas: List[float] = hp.auto(torch.optim.Adam, 'betas', ignore_docstring_errors=True)
    eps: float = hp.auto(torch.optim.Adam, 'eps', ignore_docstring_errors=True)
    weight_decay: float = hp.auto(torch.optim.Adam, 'weight_decay', ignore_docstring_errors=True)
    amsgrad: bool = hp.auto(torch.optim.Adam, 'amsgrad', ignore_docstring_errors=True)


@dataclass
class RAdamHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch_optimizer.RAdam` optimizer.

    See :class:`~torch_optimizer.RAdam` for documentation.

    Args:
        lr (float, optional): See :class:`~torch_optimizer.RAdam`.
        betas (float, optional): See :class:`~torch_optimizer.RAdam`.
        eps (float, optional): See :class:`~torch_optimizer.RAdam`.
        weight_decay (float, optional): See :class:`~torch_optimizer.RAdam`.
    """

    optimizer_cls = torch_optimizer.RAdam

    lr: float = hp.auto(torch_optimizer.RAdam, 'lr', ignore_docstring_errors=True)
    betas: List[float] = hp.auto(torch_optimizer.RAdam, 'betas', ignore_docstring_errors=True)
    eps: float = hp.auto(torch_optimizer.RAdam, 'eps', ignore_docstring_errors=True)
    weight_decay: float = hp.auto(torch_optimizer.RAdam, 'weight_decay', ignore_docstring_errors=True)


@dataclass
class AdamWHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.AdamW` optimizer.

    See :class:`~torch.optim.AdamW` for documentation.

    Args:
        lr (float, optional): See :class:`~torch.optim.AdamW`.
        betas (float, optional): See :class:`~torch.optim.AdamW`.
        eps (float, optional): See :class:`~torch.optim.AdamW`.
        weight_decay (float, optional): See :class:`~torch.optim.AdamW`.
        amsgrad (bool, optional): See :class:`~torch.optim.AdamW`.
    """

    optimizer_cls = torch.optim.AdamW

    lr: float = hp.auto(torch.optim.AdamW, 'lr', ignore_docstring_errors=True)
    betas: List[float] = hp.auto(torch.optim.AdamW, 'betas', ignore_docstring_errors=True)
    eps: float = hp.auto(torch.optim.AdamW, 'eps', ignore_docstring_errors=True)
    weight_decay: float = hp.auto(torch.optim.AdamW, 'weight_decay', ignore_docstring_errors=True)
    amsgrad: bool = hp.auto(torch.optim.AdamW, 'amsgrad', ignore_docstring_errors=True)


@dataclass
class DecoupledAdamWHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~.DecoupledAdamW` optimizer.

    See :class:`~.DecoupledAdamW` for documentation.

    Args:
        lr (float, optional): See :class:`~.DecoupledAdamW`.
        betas (float, optional): See :class:`~.DecoupledAdamW`.
        eps (float, optional): See :class:`~.DecoupledAdamW`.
        weight_decay (float, optional): See :class:`~.DecoupledAdamW`.
        amsgrad (bool, optional): See :class:`~.DecoupledAdamW`.
    """

    optimizer_cls = DecoupledAdamW

    lr: float = hp.auto(DecoupledAdamW, 'lr')
    betas: List[float] = hp.auto(DecoupledAdamW, 'betas')
    eps: float = hp.auto(DecoupledAdamW, 'eps')
    weight_decay: float = hp.auto(DecoupledAdamW, 'weight_decay')
    amsgrad: bool = hp.auto(DecoupledAdamW, 'amsgrad')


@dataclass
class SGDHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.SGD` optimizer.

    See :class:`~torch.optim.SGD` for documentation.

    Args:
        lr (float): See :class:`~torch.optim.SGD`.
        momentum (float, optional): See :class:`~torch.optim.SGD`.
        weight_decay (float, optional): See :class:`~torch.optim.SGD`.
        dampening (float, optional): See :class:`~torch.optim.SGD`.
        nesterov (bool, optional): See :class:`~torch.optim.SGD`.
    """

    optimizer_cls = torch.optim.SGD

    lr: float = hp.auto(torch.optim.SGD, 'lr', ignore_docstring_errors=True)
    momentum: float = hp.auto(torch.optim.SGD, 'momentum', ignore_docstring_errors=True)
    weight_decay: float = hp.auto(torch.optim.SGD, 'weight_decay', ignore_docstring_errors=True)
    dampening: float = hp.auto(torch.optim.SGD, 'dampening', ignore_docstring_errors=True)
    nesterov: bool = hp.auto(torch.optim.SGD, 'nesterov', ignore_docstring_errors=True)


@dataclass
class DecoupledSGDWHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~.DecoupledSGDW` optimizer.

    See :class:`~.DecoupledSGDW` for documentation.

    Args:
        lr (float): See :class:`~.DecoupledSGDW`.
        momentum (float, optional): See :class:`~.DecoupledSGDW`.
        weight_decay (float, optional): See :class:`~.DecoupledSGDW`.
        dampening (float, optional): See :class:`~.DecoupledSGDW`.
        nesterov (bool, optional): See :class:`~.DecoupledSGDW`.
    """

    optimizer_cls = DecoupledSGDW

    lr: float = hp.auto(DecoupledSGDW, 'lr')
    momentum: float = hp.auto(DecoupledSGDW, 'momentum')
    weight_decay: float = hp.auto(DecoupledSGDW, 'weight_decay')
    dampening: float = hp.auto(DecoupledSGDW, 'dampening')
    nesterov: bool = hp.auto(DecoupledSGDW, 'nesterov')


@dataclass
class RMSpropHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.RMSprop` optimizer.

    See :class:`~torch.optim.RMSprop` for documentation.

    Args:
        lr (float): See :class:`~torch.optim.RMSprop`.
        alpha (float, optional): See :class:`~torch.optim.RMSprop`.
        eps (float, optional): See :class:`~torch.optim.RMSprop`.
        momentum (float, optional): See :class:`~torch.optim.RMSprop`.
        weight_decay (float, optional): See :class:`~torch.optim.RMSprop`.
        centered (bool, optional): See :class:`~torch.optim.RMSprop`.
    """

    optimizer_cls = torch.optim.RMSprop

    lr: float = hp.auto(torch.optim.RMSprop, 'lr', ignore_docstring_errors=True)
    alpha: float = hp.auto(torch.optim.RMSprop, 'alpha', ignore_docstring_errors=True)
    eps: float = hp.auto(torch.optim.RMSprop, 'eps', ignore_docstring_errors=True)
    momentum: float = hp.auto(torch.optim.RMSprop, 'momentum', ignore_docstring_errors=True)
    weight_decay: float = hp.auto(torch.optim.RMSprop, 'weight_decay', ignore_docstring_errors=True)
    centered: float = hp.auto(torch.optim.RMSprop, 'centered', ignore_docstring_errors=True)


optimizer_registry = {
    'adam': AdamHparams,
    'adamw': AdamWHparams,
    'decoupled_adamw': DecoupledAdamWHparams,
    'radam': RAdamHparams,
    'sgd': SGDHparams,
    'decoupled_sgdw': DecoupledSGDWHparams,
    'rmsprop': RMSpropHparams,
}
