# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Optional, Sequence, Union

import torch
from torch.optim import Optimizer

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)

_TORCH_BATCHNORM_BASE_CLASS = torch.nn.modules.batchnorm._BatchNorm


def apply_ghost_batchnorm(model: torch.nn.Module,
                          ghost_batch_size: int = 32,
                          optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None) -> torch.nn.Module:
    """Replace batch normalization modules with ghost batch normalization modules.

    Ghost batch normalization modules split their input into chunks of
    ``ghost_batch_size`` samples and run batch normalization on each chunk
    separately. ``dim=0`` is assumed to be the sample axis.

    Args:
        model (torch.nn.Module): The model to modify in-place.
        ghost_batch_size (int, optional): Size of sub-batches to normalize over. Default: ``32``.
        optimizers (torch.optim.Optimizer | Sequence[torch.optim.Optimizer], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so that
            they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.

    Returns:
        The modified model

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_ghost_batchnorm(model)
    """

    def maybe_replace(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        already_ghost_batchnormed = hasattr(module, '_already_ghost_batchnormed') and module._already_ghost_batchnormed
        if isinstance(module, _TORCH_BATCHNORM_BASE_CLASS) and not already_ghost_batchnormed:
            return _GhostBatchNorm.from_batchnorm(module, ghost_batch_size=ghost_batch_size)

    # we have to specify class names explicitly because replace_module_classes
    # now checks if `module.__class__ == cls`, rather than `isinstance(module, cls)`
    transforms = {cls: maybe_replace for cls in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]}
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)
    return model


class GhostBatchNorm(Algorithm):
    """Replaces batch normalization modules with
    `Ghost Batch Normalization <https://arxiv.org/abs/1705.08741>`_ modules
    that simulate the effect of using a smaller batch size.

    Works by spliting input into chunks of ``ghost_batch_size`` samples and
    running batch normalization on each chunk separately. ``dim=0`` is assumed to
    be the sample axis.

    Runs on :attr:`.Event.INIT`.

    Args:
        ghost_batch_size (int, optional): size of sub-batches to normalize over. Default: ``32``.
    """

    def __init__(self, ghost_batch_size: int = 32):
        self.ghost_batch_size = ghost_batch_size

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(ghost_batch_size={self.ghost_batch_size})'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        assert state.model is not None, 'Model must be in state'

        apply_ghost_batchnorm(model=state.model, optimizers=state.optimizers, ghost_batch_size=self.ghost_batch_size)
        self._log_results(event, state, logger)

    def _log_results(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Logs the result of GhostBatchNorm applications, including the number of modules that have been replaced."""
        assert state.model is not None

        num_new_modules = module_surgery.count_module_instances(state.model, _GhostBatchNorm)
        classname = 'GhostBatchNorm'
        module_name = 'GhostBatchNorm'

        # python logger
        log.info(f'Applied {classname} to model {state.model.__class__.__name__} '
                 f'with ghost_batch_size={self.ghost_batch_size}, '
                 f'Model now has {num_new_modules} {module_name} modules')

        if logger is not None:
            logger.log_hyperparameters({
                f'{classname}/num_new_modules': num_new_modules,
            })


def _corresponding_ghost_batchnorm_type(batchnorm: torch.nn.Module):
    if isinstance(batchnorm, torch.nn.BatchNorm1d):
        return GhostBatchNorm1d
    if isinstance(batchnorm, torch.nn.BatchNorm2d):
        return GhostBatchNorm2d
    if isinstance(batchnorm, torch.nn.BatchNorm3d):
        return GhostBatchNorm3d
    raise ValueError(f'Input was of type {type(batchnorm)}, not one of '
                     'torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d')


class _GhostBatchNorm(torch.nn.Module):
    """`Ghost batch normalization <https://arxiv.org/abs/1705.08741>`_ layer.

    Works by spliting input into chunks of ``ghost_batch_size`` samples and
    running batch normalization on each chunk separately. ``dim=0`` is assumed to
    be the sample axis.

    See also `torch.nn.BatchNorm1d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>`_,
    `torch.nn.BatchNorm2d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`_, and
    `torch.nn.BatchNorm3d <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html>`_.

    Args:
        base_batchnorm (torch.nn.modules.batchnorm._BatchNorm): A batch normalization module to be applied to each chunk
        ghost_batch_size (int, optional): the size of the chunks passed into the underlying
            batch normalization. Default: ``32``.

    Raises:
        ValueError: If ``ghost_batch_size`` exceeds the number of samples in
            the batch provided to `forward`. This might happen when doing
            data-parallel training, because the per-worker batch size is usually
            much smaller than the overall batch size.
    """

    def __init__(self, base_batchnorm: _TORCH_BATCHNORM_BASE_CLASS, ghost_batch_size: int = 32):
        super().__init__()
        self.ghost_batch_size = ghost_batch_size
        self.batchnorm = base_batchnorm
        self.batchnorm._already_ghost_batchnormed = True  # Mark to avoid rewrapping on duplicate calls

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        batch_size = input.shape[0]

        if batch_size < self.ghost_batch_size:
            raise ValueError(f'Worker batch size {batch_size} < ghost_batch_size {self.ghost_batch_size}')

        nchunks: int = int(math.ceil(batch_size / self.ghost_batch_size))
        has_momentum = self.batchnorm.momentum is not None
        original_momentum: float = self.batchnorm.momentum

        if self.training and has_momentum:
            # applying the same batchnorm multiple times greatly increases
            # the variance of the moving average statistics; reduce the
            # exponential moving average constant proportionally
            # to compensate.
            self._scale_momentum(nchunks)

        normalized_chunks = [self.batchnorm(chunk) for chunk in input.chunk(nchunks, 0)]

        if self.training and has_momentum:
            self._unscale_momentum(original_momentum)

        return torch.cat(normalized_chunks, dim=0)

    @staticmethod
    def from_batchnorm(module: torch.nn.Module, ghost_batch_size: int) -> '_GhostBatchNorm':
        assert isinstance(module, _TORCH_BATCHNORM_BASE_CLASS), 'Module is not a BatchNorm subclass!'
        bn_type = _corresponding_ghost_batchnorm_type(module)
        return bn_type(ghost_batch_size=ghost_batch_size, base_batchnorm=module)

    @torch.jit.unused
    def _scale_momentum(self, nchunks: int):
        self.batchnorm.momentum = float(self.batchnorm.momentum) / nchunks

    @torch.jit.unused
    def _unscale_momentum(self, original_momentum: float):
        self.batchnorm.momentum = original_momentum


class GhostBatchNorm1d(_GhostBatchNorm):
    pass


class GhostBatchNorm2d(_GhostBatchNorm):
    pass


class GhostBatchNorm3d(_GhostBatchNorm):
    pass
