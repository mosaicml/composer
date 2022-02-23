# Copyright 2021 MosaicML. All Rights Reserved.

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Type

import torch
import torch_optimizer
import yahp as hp

from composer.core.types import ModelParameters, Optimizer
from composer.optim import DecoupledAdamW, DecoupledSGDW

# Optimizer parameters and defaults match those in torch.optim


@dataclass
class OptimizerHparams(hp.Hparams, ABC):
    """Abstract base class for optimizer hyperparameter classes."""

    @property
    @abstractmethod
    def optimizer_object(cls) -> Type[Optimizer]:
        pass

    def initialize_object(self, param_group: ModelParameters) -> Optimizer:
        assert issubclass(self.optimizer_object, torch.optim.Optimizer)
        return self.optimizer_object(param_group, **asdict(self))


@dataclass
class AdamHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.Adam` optimizer."""
    lr: float = hp.optional(default=0.001, doc='learning rate')
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc='coefficients used for computing running averages of gradient and its square.')
    eps: float = hp.optional(default=1e-8, doc='term for numerical stability')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')
    amsgrad: bool = hp.optional(default=False, doc='use AMSGrad variant')

    @property
    def optimizer_object(cls) -> Type[torch.optim.Adam]:
        return torch.optim.Adam


@dataclass
class RAdamHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.RAdam` optimizer."""
    lr: float = hp.optional(default=0.001, doc='learning rate')
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc='coefficients used for computing running averages of gradient and its square.')
    eps: float = hp.optional(default=1e-8, doc='term for numerical stability')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')

    @property
    def optimizer_object(cls) -> Type[torch_optimizer.RAdam]:
        return torch_optimizer.RAdam


@dataclass
class AdamWHparams(OptimizerHparams):
    """Hyperparameters for the :class:`torch.optim.AdamW` optimizer."""
    lr: float = hp.optional(default=0.001, doc='learning rate')
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc='coefficients used for computing running averages of gradient and its square.')
    eps: float = hp.optional(default=1e-8, doc='term for numerical stability')
    weight_decay: float = hp.optional(default=1e-2, doc='weight decay (L2 penalty)')
    amsgrad: bool = hp.optional(default=False, doc='use AMSGrad variant')

    @property
    def optimizer_object(cls) -> Type[torch.optim.AdamW]:
        return torch.optim.AdamW


@dataclass
class DecoupledAdamWHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~composer.optim.DecoupledAdamW` optimizer."""
    lr: float = hp.optional(default=0.001, doc='learning rate')
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc='coefficients used for computing running averages of gradient and its square.')
    eps: float = hp.optional(default=1e-8, doc='term for numerical stability')
    weight_decay: float = hp.optional(default=1e-2, doc='weight decay (L2 penalty)')
    amsgrad: bool = hp.optional(default=False, doc='use AMSGrad variant')

    @property
    def optimizer_object(cls) -> Type[DecoupledAdamW]:
        return DecoupledAdamW


@dataclass
class SGDHparams(OptimizerHparams):
    """Hyperparameters for the `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD>`_
    optimizer."""
    lr: float = hp.required(doc='learning rate')
    momentum: float = hp.optional(default=0.0, doc='momentum factor')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')
    dampening: float = hp.optional(default=0.0, doc='dampening for momentum')
    nesterov: bool = hp.optional(default=False, doc='Nesterov momentum')

    @property
    def optimizer_object(cls) -> Type[torch.optim.SGD]:
        return torch.optim.SGD


@dataclass
class DecoupledSGDWHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~composer.optim.DecoupledSGDW` optimizer."""
    lr: float = hp.required(doc='learning rate')
    momentum: float = hp.optional(default=0.0, doc='momentum factor')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')
    dampening: float = hp.optional(default=0.0, doc='dampening for momentum')
    nesterov: bool = hp.optional(default=False, doc='Nesterov momentum')

    @property
    def optimizer_object(cls) -> Type[DecoupledSGDW]:
        return DecoupledSGDW


@dataclass
class RMSPropHparams(OptimizerHparams):
    """Hyperparameters for the [RMSProp
    optimizer](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop)."""
    lr: float = hp.required(doc='learning rate')
    alpha: float = hp.optional(default=0.99, doc='smoothing constant')
    eps: float = hp.optional(default=1e-8, doc='term for numerical stability')
    momentum: float = hp.optional(default=0.0, doc='momentum factor')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')
    centered: bool = hp.optional(
        default=False,
        doc='normalize gradient by an estimation of variance',
    )

    @property
    def optimizer_object(cls) -> Type[torch.optim.RMSprop]:
        return torch.optim.RMSprop


def get_optimizer(param_groups: ModelParameters, hparams: OptimizerHparams) -> Optimizer:
    """Get the optimizer specified by the given hyperparameters.

    Args:
        param_groups (ModelParameters): List of model parameters to optimize.
        hparams (OptimizerHparams): Instance of an optimizer's hyperparameters.
    """

    return hparams.initialize_object(param_group=param_groups)
