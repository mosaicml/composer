# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional, TypeVar

import numpy as np
import torch
import torch.nn.functional as F
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery
from composer.core.types import Optimizers

log = logging.getLogger(__name__)

_DEFAULT_GHOST_BATCH_SIZE = 32
_TORCH_BATCHNORM_BASE_CLASS = torch.nn.modules.batchnorm._BatchNorm


def _corresponding_ghost_batchnorm_type(batchnorm: torch.nn.Module):
    if isinstance(batchnorm, torch.nn.BatchNorm1d):
        return GhostBatchNorm1d
    if isinstance(batchnorm, torch.nn.BatchNorm2d):
        return GhostBatchNorm2d
    if isinstance(batchnorm, torch.nn.BatchNorm3d):
        return GhostBatchNorm3d
    raise ValueError(f"Input was of type {type(batchnorm)}, not one of "
                     "torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d")


T = TypeVar('T', bound='_GhostBatchNorm')

# class _GhostBatchNorm(torch.nn.Module):
class _GhostBatchNorm(torch.nn.batchnorm._BatchNorm):
    """`Ghost batch normalization <https://arxiv.org/abs/1705.08741>`_ layer.

    Works by spliting input into chunks of ``ghost_batch_size`` samples and
    running batch normalization on each chunk separately. Dim 0 is assumed to
    be the sample axis.

    See also `torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>`_,  `torch.nn.BatchNorm2d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_, and
    `torch.nn.BatchNorm3d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html>`_.

    Args:
        ghost_batch_size: the size of the chunks passed into the underlying
            batch normalization
        base_batchnorm: A batch normalization module to be applied to each chunk

    Raises:
        ValueError: If ``ghost_batch_size`` exceeds the number of samples in
            the batch provided to `forward`. This might happen when doing
            data-parallel training, because the per-worker batch size is usually
            much smaller than the overall batch size.
    """
    # def __init__(self, num_features: int, ghost_batch_size: int = _DEFAULT_GHOST_BATCH_SIZE,
    #              affine: bool = True, track_running_stats: bool = True, ) -> None:
    def __init__(self,
                 num_features: int,
                 ghost_batch_size: int = _DEFAULT_GHOST_BATCH_SIZE,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 device=None,
                 dtype=None):
        super().__init__(num_features=num_features,
                         eps=eps,
                         momentum=momentum,
                         affine=affine,
                         track_running_stats=track_running_stats,
                         device=device,
                         dtype=dtype)
        # self.num_features = num_features
        self.ghost_batch_size = ghost_batch_size
        # self.affine = affine
        # self.track_running_stats = track_running_stats

        # if track_running_stats:
        #     self.register_buffer('running_mean', torch.zeros(self.num_features))
        #     self.register_buffer('running_var', torch.ones(self.num_features))
        # else:


    # def _has_momentum(self) -> bool:
    #     return hasattr(self.batchnorm, 'momentum') and self.batchnorm.momentum is not None

    # def train(self: T, mode: bool = True) -> T:

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        if not self.training:  # for inference, behave like a normal batchnorm
            return super().forward(input)

        # The exponential average logic below is taken verbatim from torch batchnorm
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # from here down is (mostly) new logic; we reshape the tensor so that
        # different samples are treated as different channels, then call a
        # regular batchnorm function, and then aggregate the running stats
        # from all the corresponding channels
        batch_size = input.shape[0]
        has_stragglers = batch_size % self.ghost_batch_size != 0
        if has_stragglers:
            raise ValueError(f"Ghost batch size {self.ghost_batch_size} does not evenly divide per-device batch size {batch_size}")
        num_ghost_batches = batch_size // self.ghost_batch_size

        strides = input.stride()
        is_nchw = smallest_stride = np.min(strides)

        running_mean, running_var = None, None
        if self.track_running_stats:
            running_mean = self.running_mean
            running_var = self.running_var
            if is_nchw:
                running_mean.repeat(num_ghost_batches)
                running_var.repeat(num_ghost_batches)

        if not is_nchw:
            input = input.to(memory_format=torch.contiguous_format)
        X = input.reshape(self.ghost_batch_size, num_ghost_batches * input.shape[1], *input.shape[2:])

        # if is_nchw:  # nchw layout or equivalent lets us use F.batch_norm
        #     X = input.reshape(self.ghost_batch_size, num_ghost_batches * input.shape[1], *input.shape[2:])
        # elif strides[-1] == smallest_stride:

            # # reshaping yields the wrong layout, so do the batchnorm ourselves
            # X = input.reshape(self.ghost_batch_size, num_ghost_batches, *input.shape[2:])
            # # e.g., for nchw -> g, n/g, c, h, w
            # dim = [1] + list(range(1, X.ndim))
            # var, mean = torch.var_mean(X, dim=dim, unbiased=False, keepdim=True)
            # X = X - mean
            # X /= var
            # if self.track_running_stats and self.exponential_average_factor:
            #     self.running_mean


        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        ret = F.batch_norm(
            X,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,  # whether training; always True if we got to here
            exponential_average_factor,
            self.eps,
        )
        if self.track_running_stats and num_ghost_batches > 1:
            # average stats from all ghost batches; the tensors we passed
            # in were updated in place by F.batch_norm
            self.running_mean.copy_(running_mean.reshape(num_ghost_batches, self.num_features)).mean(dim=0)
            self.running_var.copy_(running_mean.reshape(num_ghost_batches, self.num_features)).mean(dim=0)
        if not is_nchw:  # XXX tensor could be in a format other than NCHW or NHWC
            ret = ret.to(memory_format=torch.channels_last)
        return ret.reshape(input.shape)


    @staticmethod
    def from_batchnorm(module: torch.nn.Module, ghost_batch_size: int) -> _GhostBatchNorm:
        assert isinstance(module, _TORCH_BATCHNORM_BASE_CLASS), "Module is not a BatchNorm subclass!"
        bn_type = _corresponding_ghost_batchnorm_type(module)
        return bn_type(ghost_batch_size=ghost_batch_size, base_batchnorm=module)


class GhostBatchNorm1d(_GhostBatchNorm):

    def _check_input_dim(self, input):  # copied from torch BatchNorm1d
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("expected 2D or 3D input (got {}D input)".format(input.dim()))


class GhostBatchNorm2d(_GhostBatchNorm):

    def _check_input_dim(self, input):  # copied from torch BatchNorm2d
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class GhostBatchNorm3d(_GhostBatchNorm):

    def _check_input_dim(self, input):  # copied from torch BatchNorm3d
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


def apply_ghost_batchnorm(model: torch.nn.Module,
                          ghost_batch_size: int,
                          optimizers: Optional[Optimizers] = None) -> torch.nn.Module:
    """Replace batch normalization modules with ghost batch normalization modules.

    Must be run before the model has been moved to accelerators and before
    the model's parameters have been passed to an optimizer.

    Args:
        model: model to transform
        ghost_batch_size: size of sub-batches to normalize over
        optimizers (Optimizers, optional):  Existing optimizers bound to ``model.parameters()``.
            All optimizers that have already been constructed with,
            ``model.parameters()`` must be specified here so they will optimize
            the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.
    """

    def maybe_replace(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        if isinstance(module, _TORCH_BATCHNORM_BASE_CLASS):
            return _GhostBatchNorm.from_batchnorm(module, ghost_batch_size=ghost_batch_size)

    # we have to specify class names explicitly because replace_module_classes
    # now checks if `module.__class__ == cls`, rather than `isinstance(module, cls)`
    transforms = {cls: maybe_replace for cls in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]}
    surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)
    return model


@dataclass
class GhostBatchNormHparams(AlgorithmHparams):
    """See :class:`GhostBatchNorm`"""

    ghost_batch_size: int = hp.required(doc='Size of sub-batches to normalize over',
                                        template_default=_DEFAULT_GHOST_BATCH_SIZE)

    def initialize_object(self) -> "GhostBatchNorm":
        return GhostBatchNorm(**asdict(self))


class GhostBatchNorm(Algorithm):
    """Replaces batch normalization modules with `Ghost Batch Normalization <https://arxiv.org/abs/1705.08741>`_ modules
    that simulate the effect of using a smaller batch size.

    Works by spliting input into chunks of ``ghost_batch_size`` samples and
    running batch normalization on each chunk separately. Dim 0 is assumed to
    be the sample axis.

    Runs on ``Event.INIT`` and should be applied both before the model has
    been moved to accelerators and before the model’s parameters have
    been passed to an optimizer.

    Args:
        ghost_batch_size: size of sub-batches to normalize over
    """

    def __init__(self, ghost_batch_size: int = _DEFAULT_GHOST_BATCH_SIZE):
        self.ghost_batch_size = ghost_batch_size

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.INIT."""
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Applies GhostBatchNorm by wrapping existing BatchNorm modules."""
        assert state.model is not None, "Model must be in state"

        apply_ghost_batchnorm(model=state.model, optimizers=state.optimizers, ghost_batch_size=self.ghost_batch_size)
        self._log_results(event, state, logger)

    def _log_results(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Logs the result of GhostBatchNorm applications, including the number of modules that have been replaced."""
        assert state.model is not None

        num_new_modules = surgery.count_module_instances(state.model, _GhostBatchNorm)
        classname = 'GhostBatchNorm'
        module_name = 'GhostBatchNorm'

        # python logger
        log.info(f'Applied {classname} to model {state.model.__class__.__name__} '
                 f'with ghost_batch_size={self.ghost_batch_size}, '
                 f'Model now has {num_new_modules} {module_name} modules')

        if logger is not None:
            logger.metric_fit({
                f'{classname}/num_new_modules': num_new_modules,
            })
