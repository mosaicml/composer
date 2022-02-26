# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import functools
import logging
from typing import Optional

import numpy as np
import torch

from composer.algorithms.blurpool.blurpool_layers import BlurConv2d, BlurMaxPool2d
from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Optimizers
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def apply_blurpool(model: torch.nn.Module,
                   replace_convs: bool = True,
                   replace_maxpools: bool = True,
                   blur_first: bool = True,
                   optimizers: Optional[Optimizers] = None) -> torch.nn.Module:
    """Add anti-aliasing filters to the strided :class:`torch.nn.Conv2d` and/or :class:`torch.nn.MaxPool2d` modules
    within `model`.

    These filters increase invariance to small spatial shifts in the input
    (`Zhang 2019 <http://proceedings.mlr.press/v97/zhang19a.html>`_).

    Args:
        model (torch.nn.Module): the model to modify in-place
        replace_convs (bool, optional): replace strided :class:`torch.nn.Conv2d` modules with
            :class:`.BlurConv2d` modules
        replace_maxpools (bool, optional): replace eligible :class:`torch.nn.MaxPool2d` modules
            with :class:`.BlurMaxPool2d` modules.
        blur_first (bool, optional): for ``replace_convs``, blur input before the associated
            convolution. When set to ``False``, the convolution is applied with
            a stride of 1 before the blurring, resulting in significant
            overhead (though more closely matching
            `the paper <http://proceedings.mlr.press/v97/zhang19a.html>`_).
            See :class:`.BlurConv2d` for further discussion.
        optimizers (Optimizers, optional):  Existing optimizers bound to
            ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so
            they will optimize the correct parameters.

            If the optimizer(s) are constructed *after* calling this function,
            then it is safe to omit this parameter. These optimizers will see
            the correct model parameters.

    Returns:
        The modified model

    Example:
        .. testcode::

            import composer.functional as cf
            from torchvision import models
            model = models.resnet50()
            cf.apply_blurpool(model)
    """
    transforms = {}
    if replace_maxpools:
        transforms[torch.nn.MaxPool2d] = BlurMaxPool2d.from_maxpool2d
    if replace_convs:
        transforms[torch.nn.Conv2d] = functools.partial(
            _maybe_replace_strided_conv2d,
            blur_first=blur_first,
        )
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)
    _log_surgery_result(model)

    return model


class BlurPool(Algorithm):
    """`BlurPool <http://proceedings.mlr.press/v97/zhang19a.html>`_ adds anti-aliasing filters to convolutional layers
    to increase accuracy and invariance to small shifts in the input.

    Runs on :attr:`~composer.core.event.Event.INIT`.

    Args:
        replace_convs (bool): replace strided :class:`torch.nn.Conv2d` modules with
            :class:`.BlurConv2d` modules
        replace_maxpools (bool): replace eligible :class:`torch.nn.MaxPool2d` modules
            with :class:`.BlurMaxPool2d` modules.
        blur_first (bool): when ``replace_convs`` is ``True``, blur input before the
            associated convolution. When set to ``False``, the convolution is
            applied with a stride of 1 before the blurring, resulting in
            significant overhead (though more closely matching the paper).
            See :class:`.BlurConv2d` for further discussion.
    """

    def __init__(self, replace_convs: bool, replace_maxpools: bool, blur_first: bool) -> None:
        self.replace_convs = replace_convs
        self.replace_maxpools = replace_maxpools
        self.blur_first = blur_first

        if self.replace_maxpools is False and \
             self.replace_convs is False:
            log.warning('Both replace_maxpool and replace_convs set to false '
                        'BlurPool will not be modifying the model.')

    def match(self, event: Event, state: State) -> bool:
        """Runs on :attr:`~composer.core.event.Event.INIT`.

        Args:
            event (Event): The current event.
            state (State): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Adds anti-aliasing filters to the maxpools and/or convolutions.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None

        apply_blurpool(state.model,
                       optimizers=state.optimizers,
                       replace_convs=self.replace_convs,
                       replace_maxpools=self.replace_maxpools,
                       blur_first=self.blur_first)
        self._log_results(event, state, logger)

    def _log_results(self, event: Event, state: State, logger: Logger) -> None:
        """Logs the result of BlurPool application, including the number of layers that have been replaced."""
        assert state.model is not None

        num_blurpool_layers = module_surgery.count_module_instances(state.model, BlurMaxPool2d)
        num_blurconv_layers = module_surgery.count_module_instances(state.model, BlurConv2d)

        # python logger
        log.info(f'Applied BlurPool to model {state.model.__class__.__name__} '
                 f'with replace_maxpools={self.replace_maxpools}, '
                 f'replace_convs={self.replace_convs}. '
                 f'Model now has {num_blurpool_layers} BlurMaxPool2d '
                 f'and {num_blurconv_layers} BlurConv2D layers.')

        logger.metric_fit({
            'blurpool/num_blurpool_layers': num_blurpool_layers,
            'blurpool/num_blurconv_layers': num_blurconv_layers,
        })


def _log_surgery_result(model: torch.nn.Module):
    num_blurpool_layers = module_surgery.count_module_instances(model, BlurMaxPool2d)
    num_blurconv_layers = module_surgery.count_module_instances(model, BlurConv2d)
    log.info(f'Applied BlurPool to model {model.__class__.__name__}. '
             f'Model now has {num_blurpool_layers} BlurMaxPool2d '
             f'and {num_blurconv_layers} BlurConv2D layers.')


def _maybe_replace_strided_conv2d(module: torch.nn.Conv2d, module_index: int, blur_first: bool):
    if (np.max(module.stride) > 1 and module.in_channels >= 16):
        return BlurConv2d.from_conv2d(module, module_index, blur_first=blur_first)
    return None
