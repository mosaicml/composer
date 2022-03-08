# Copyright 2021 MosaicML. All Rights Reserved.

"""Hyperparameters for optimizers."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Type

import torch
import torch_optimizer
import yahp as hp

from composer.core.types import ModelParameters, Optimizer
from composer.optim import DecoupledAdamW, DecoupledSGDW

# Optimizer parameters and defaults match those in torch.optim

__all__ = [
    "OptimizerHparams", "AdamHparams", "RAdamHparams", "AdamWHparams", "DecoupledAdamWHparams", "SGDHparams",
    "DecoupledSGDWHparams", "RMSpropHparams"
]


@dataclass
class OptimizerHparams(hp.Hparams, ABC):
    """Base class for optimizer hyperparameter classes.

    Optimizer parameters that are added to :class:`~composer.trainer.trainer_hparams.TrainerHparams` (e.g. via YAML or
    the CLI) are initialized in the training loop.
    """

    @property
    @abstractmethod
    def optimizer_object(cls) -> Type[Optimizer]:
        pass

    def initialize_object(self, param_group: ModelParameters) -> Optimizer:
        """Initializes the optimizer.

        Args:
            param_group (:attr:`~composer.core.types.ModelParameters`): Parameters for this optimizer to optimize.
        """

        assert issubclass(self.optimizer_object, torch.optim.Optimizer)
        return self.optimizer_object(param_group, **asdict(self))


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
    lr: float = hp.optional(default=0.001, doc="learning rate")
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc="coefficients used for computing running averages of gradient and its square.")
    eps: float = hp.optional(default=1e-8, doc="term for numerical stability")
    weight_decay: float = hp.optional(default=0.0, doc="weight decay (L2 penalty)")
    amsgrad: bool = hp.optional(default=False, doc="use AMSGrad variant")

    @property
    def optimizer_object(cls) -> Type[torch.optim.Adam]:
        return torch.optim.Adam


@dataclass
class RAdamHparams(OptimizerHparams):
    """Hyperparameters for the :class:`~torch.optim.RAdam` optimizer.

    See :class:`~torch.optim.RAdam` for documentation.

    Args:
        lr (float, optional): See :class:`~torch.optim.RAdam`.
        betas (float, optional): See :class:`~torch.optim.RAdam`.
        eps (float, optional): See :class:`~torch.optim.RAdam`.
        weight_decay (float, optional): See :class:`~torch.optim.RAdam`.
    """
    lr: float = hp.optional(default=0.001, doc="learning rate")
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc="coefficients used for computing running averages of gradient and its square.")
    eps: float = hp.optional(default=1e-8, doc="term for numerical stability")
    weight_decay: float = hp.optional(default=0.0, doc="weight decay (L2 penalty)")

    @property
    def optimizer_object(cls) -> Type[torch_optimizer.RAdam]:
        return torch_optimizer.RAdam


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
    lr: float = hp.optional(default=0.001, doc="learning rate")
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc="coefficients used for computing running averages of gradient and its square.")
    eps: float = hp.optional(default=1e-8, doc="term for numerical stability")
    weight_decay: float = hp.optional(default=1e-2, doc="weight decay (L2 penalty)")
    amsgrad: bool = hp.optional(default=False, doc="use AMSGrad variant")

    @property
    def optimizer_object(cls) -> Type[torch.optim.AdamW]:
        return torch.optim.AdamW


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
    lr: float = hp.optional(default=0.001, doc="learning rate")
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc="coefficients used for computing running averages of gradient and its square.")
    eps: float = hp.optional(default=1e-8, doc="term for numerical stability")
    weight_decay: float = hp.optional(default=1e-2, doc="weight decay (L2 penalty)")
    amsgrad: bool = hp.optional(default=False, doc="use AMSGrad variant")

    @property
    def optimizer_object(cls) -> Type[DecoupledAdamW]:
        return DecoupledAdamW


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
    lr: float = hp.required(doc="learning rate")
    momentum: float = hp.optional(default=0.0, doc="momentum factor")
    weight_decay: float = hp.optional(default=0.0, doc="weight decay (L2 penalty)")
    dampening: float = hp.optional(default=0.0, doc="dampening for momentum")
    nesterov: bool = hp.optional(default=False, doc="Nesterov momentum")

    @property
    def optimizer_object(cls) -> Type[torch.optim.SGD]:
        return torch.optim.SGD


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
    lr: float = hp.required(doc="learning rate")
    momentum: float = hp.optional(default=0.0, doc="momentum factor")
    weight_decay: float = hp.optional(default=0.0, doc="weight decay (L2 penalty)")
    dampening: float = hp.optional(default=0.0, doc="dampening for momentum")
    nesterov: bool = hp.optional(default=False, doc="Nesterov momentum")

    @property
    def optimizer_object(cls) -> Type[DecoupledSGDW]:
        return DecoupledSGDW


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
    lr: float = hp.required(doc="learning rate")
    alpha: float = hp.optional(default=0.99, doc="smoothing constant")
    eps: float = hp.optional(default=1e-8, doc="term for numerical stability")
    momentum: float = hp.optional(default=0.0, doc="momentum factor")
    weight_decay: float = hp.optional(default=0.0, doc="weight decay (L2 penalty)")
    centered: bool = hp.optional(
        default=False,
        doc="normalize gradient by an estimation of variance",
    )

    @property
    def optimizer_object(cls) -> Type[torch.optim.RMSprop]:
        return torch.optim.RMSprop
