# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.core import Algorithm, Event, Logger, State, surgery

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


class _GhostBatchNorm(torch.nn.Module):
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

    def __init__(self, base_batchnorm: _TORCH_BATCHNORM_BASE_CLASS, ghost_batch_size: int = _DEFAULT_GHOST_BATCH_SIZE):
        super().__init__()
        self.ghost_batch_size = ghost_batch_size
        self.batchnorm = base_batchnorm

    def _has_momentum(self) -> bool:
        return hasattr(self.batchnorm, 'momentum') and self.batchnorm.momentum is not None

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        batch_size = input.shape[0]
        if batch_size < self.ghost_batch_size:
            raise ValueError(f"Worker batch size {batch_size} < ghost_batch_size {self.ghost_batch_size}")

        nchunks = int(np.ceil(batch_size / self.ghost_batch_size))
        has_momentum = self._has_momentum()
        if has_momentum:
            # applying the same batchnorm multiple times greatly increases
            # the variance of the moving average statistics; reduce the
            # exponential moving average constant proportionally
            # to partially compensate for this
            original_momentum = self.batchnorm.momentum
            self.batchnorm.momentum = float(original_momentum) / nchunks  # type: ignore
        normalized_chunks = [self.batchnorm(chunk) for chunk in input.chunk(nchunks, 0)]
        if has_momentum:
            self.batchnorm.momentum = original_momentum  # type: ignore

        return torch.cat(normalized_chunks, dim=0)

    @staticmethod
    def from_batchnorm(module: torch.nn.Module, ghost_batch_size: int) -> _GhostBatchNorm:
        assert isinstance(module, _TORCH_BATCHNORM_BASE_CLASS), "Module is not a BatchNorm subclass!"
        bn_type = _corresponding_ghost_batchnorm_type(module)
        return bn_type(ghost_batch_size=ghost_batch_size, base_batchnorm=module)


class GhostBatchNorm1d(_GhostBatchNorm):
    pass


class GhostBatchNorm2d(_GhostBatchNorm):
    pass


class GhostBatchNorm3d(_GhostBatchNorm):
    pass


def apply_ghost_batchnorm(model: torch.nn.Module, ghost_batch_size: int) -> torch.nn.Module:
    """Replace batch normalization modules with ghost batch normalization modules.

    Must be run before the model has been moved to accelerators and before
    the model's parameters have been passed to an optimizer.

    Args:
        model: model to transform
        ghost_batch_size: size of sub-batches to normalize over
    """

    def maybe_replace(module: torch.nn.Module, module_index: int) -> Optional[torch.nn.Module]:
        if isinstance(module, _TORCH_BATCHNORM_BASE_CLASS):
            return _GhostBatchNorm.from_batchnorm(module, ghost_batch_size=ghost_batch_size)

    # we have to specify class names explicitly because replace_module_classes
    # now checks if `module.__class__ == cls`, rather than `isinstance(module, cls)`
    transforms = {cls: maybe_replace for cls in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d]}
    surgery.replace_module_classes(model, policies=transforms)
    return model


@dataclass
class GhostBatchNormHparams(AlgorithmHparams):
    """See :class:`GhostBatchNorm`"""

    ghost_batch_size: int = hp.required(doc='Size of sub-batches to normalize over',
                                        template_default=_DEFAULT_GHOST_BATCH_SIZE)

    def initialize_object(self) -> "GhostBatchNorm":
        return GhostBatchNorm(**asdict(self))


class GhostBatchNorm(Algorithm):
    """Replaces batch normalization modules with
    `Ghost Batch Normalization <https://arxiv.org/abs/1705.08741>`_ modules
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
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """ Applies GhostBatchNorm by wrapping existing BatchNorm modules
        """
        assert state.model is not None, "Model must be in state"

        apply_ghost_batchnorm(model=state.model, ghost_batch_size=self.ghost_batch_size)
        self._log_results(event, state, logger)

    def _log_results(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Logs the result of GhostBatchNorm applications, including the number
        of modules that have been replaced.
        """
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
