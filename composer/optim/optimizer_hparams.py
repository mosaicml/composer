from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import List, Type

import torch
import torch_optimizer
import yahp as hp

from composer.core.types import ModelParameters, Optimizer
from composer.optim import MosaicMLAdamW, MosaicMLSGDW

# Optimizer parameters and defaults match those in torch.optim


@dataclass
class OptimizerHparams(hp.Hparams, ABC):

    @property
    @abstractmethod
    def optimizer_object(cls) -> Type[Optimizer]:
        pass

    def initialize_object(self, param_group: ModelParameters) -> Optimizer:
        assert issubclass(self.optimizer_object, torch.optim.Optimizer)
        return self.optimizer_object(param_group, **asdict(self))


@dataclass
class AdamHparams(OptimizerHparams):
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
class MosaicMLAdamWHparams(OptimizerHparams):
    lr: float = hp.optional(default=0.001, doc='learning rate')
    betas: List[float] = hp.optional(default_factory=lambda: [0.9, 0.999],
                                     doc='coefficients used for computing running averages of gradient and its square.')
    eps: float = hp.optional(default=1e-8, doc='term for numerical stability')
    weight_decay: float = hp.optional(default=1e-2, doc='weight decay (L2 penalty)')
    amsgrad: bool = hp.optional(default=False, doc='use AMSGrad variant')

    @property
    def optimizer_object(cls) -> Type[MosaicMLAdamW]:
        return MosaicMLAdamW


@dataclass
class SGDHparams(OptimizerHparams):
    lr: float = hp.required(doc='learning rate')
    momentum: float = hp.optional(default=0.0, doc='momentum factor')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')
    dampening: float = hp.optional(default=0.0, doc='dampening for momentum')
    nesterov: bool = hp.optional(default=False, doc='Nesterov momentum')

    @property
    def optimizer_object(cls) -> Type[torch.optim.SGD]:
        return torch.optim.SGD


@dataclass
class MosaicMLSGDWHparams(OptimizerHparams):
    lr: float = hp.required(doc='learning rate')
    momentum: float = hp.optional(default=0.0, doc='momentum factor')
    weight_decay: float = hp.optional(default=0.0, doc='weight decay (L2 penalty)')
    dampening: float = hp.optional(default=0.0, doc='dampening for momentum')
    nesterov: bool = hp.optional(default=False, doc='Nesterov momentum')

    @property
    def optimizer_object(cls) -> Type[MosaicMLSGDW]:
        return MosaicMLSGDW


@dataclass
class RMSPropHparams(OptimizerHparams):
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


def get_optimizer(param_groups: ModelParameters, hparams: OptimizerHparams):
    return hparams.initialize_object(param_group=param_groups)
