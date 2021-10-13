# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

import functools
import logging
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import yahp as hp

from composer.algorithms import AlgorithmHparams
from composer.algorithms.blurpool.blurpool_layers import BlurConv2d, BlurMaxPool2d
from composer.core import Algorithm, Event, Logger, State, surgery

log = logging.getLogger(__name__)


def _log_surgery_result(model: torch.nn.Module):
    num_blurpool_layers = surgery.count_module_instances(model, BlurMaxPool2d)
    num_blurconv_layers = surgery.count_module_instances(model, BlurConv2d)
    log.info(f'Applied BlurPool to model {model.__class__.__name__}. '
             f'Model now has {num_blurpool_layers} BlurMaxPool2d '
             f'and {num_blurconv_layers} BlurConv2D layers.')


def apply_blurpool(model: torch.nn.Module,
                   replace_convs: bool = True,
                   replace_maxpools: bool = True,
                   blur_first: bool = True) -> None:
    """Add anti-aliasing filters to the strided :class:`torch.nn.Conv2d`
    and/or :class:`torch.nn.MaxPool2d` modules within `model`.

    Must be run before the model has been moved to accelerators and before
    the model's parameters have been passed to an optimizer.

    Args:
        model: model to modify
        replace_convs: replace strided :class:`torch.nn.Conv2d` modules with
            :class:`BlurConv2d` modules
        replace_maxpools: replace eligible :class:`torch.nn.MaxPool2d` modules
            with :class:`BlurMaxPool2d` modules.
        blur_first: for ``replace_convs``, blur input before the associated
            convolution. When set to ``False``, the convolution is applied with
            a stride of 1 before the blurring, resulting in significant
            overhead (though more closely matching
            `the paper <http://proceedings.mlr.press/v97/zhang19a.html>`_).
            See :class:`~composer.algorithms.blurpool.BlurConv2d` for further discussion.
    """
    transforms = {}
    if replace_maxpools:
        transforms[torch.nn.MaxPool2d] = BlurMaxPool2d.from_maxpool2d
    if replace_convs:
        transforms[torch.nn.Conv2d] = functools.partial(
            _maybe_replace_strided_conv2d,
            blur_first=blur_first,
        )
    surgery.replace_module_classes(model, policies=transforms)
    _log_surgery_result(model)


@dataclass
class BlurPoolHparams(AlgorithmHparams):
    """See :class:`BlurPool`"""

    replace_convs: bool = hp.optional('Replace Conv2d with BlurConv2d if stride > 1', default=True)
    replace_maxpools: bool = hp.optional('Replace MaxPool2d with BlurMaxPool2d', default=True)
    blur_first: bool = hp.optional('Blur input before convolution', default=True)

    def initialize_object(self) -> "BlurPool":
        return BlurPool(**asdict(self))


def _maybe_replace_strided_conv2d(module: torch.nn.Conv2d, module_index: int, blur_first: bool):
    if (np.max(module.stride) > 1 and module.in_channels >= 16):
        return BlurConv2d.from_conv2d(module, module_index, blur_first=blur_first)
    return None


class BlurPool(Algorithm):
    """`BlurPool <http://proceedings.mlr.press/v97/zhang19a.html>`_
    adds anti-aliasing filters to convolutional layers to increase accuracy
    and invariance to small shifts in the input.

    Runs on ``Event.INIT`` and should be applied both before the model has
    been moved to accelerators and before the model’s parameters have
    been passed to an optimizer.

    Args:
        replace_convs: replace strided :class:`torch.nn.Conv2d` modules with
            :class:`BlurConv2d` modules
        replace_maxpools: replace eligible :class:`torch.nn.MaxPool2d` modules
            with :class:`BlurMaxPool2d` modules.
        blur_first: when ``replace_convs`` is ``True``, blur input before the
            associated convolution. When set to ``False``, the convolution is
            applied with a stride of 1 before the blurring, resulting in
            significant overhead (though more closely matching the paper).
            See :class:`~composer.algorithms.blurpool.BlurConv2d` for further discussion.
    """

    def __init__(self, replace_convs: bool, replace_maxpools: bool, blur_first: bool) -> None:
        self.hparams = BlurPoolHparams(
            replace_convs=replace_convs,
            replace_maxpools=replace_maxpools,
            blur_first=blur_first,
        )

        if self.hparams.replace_maxpools is False and \
             self.hparams.replace_convs is False:
            log.warning('Both replace_maxpool and replace_convs set to false '
                        'BlurPool will not be modifying the model.')

    def match(self, event: Event, state: State) -> bool:
        """Runs on Event.INIT
        
        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now.
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """Adds anti-aliasing filters to the maxpools and/or convolutions
        
        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        assert state.model is not None

        apply_blurpool(state.model, **asdict(self.hparams))
        self._log_results(event, state, logger)

    def _log_results(self, event: Event, state: State, logger: Logger) -> None:
        """ Logs the result of BlurPool application, including the number
        of layers that have been replaced.
        """
        assert state.model is not None

        num_blurpool_layers = surgery.count_module_instances(state.model, BlurMaxPool2d)
        num_blurconv_layers = surgery.count_module_instances(state.model, BlurConv2d)

        # python logger
        log.info(f'Applied BlurPool to model {state.model.__class__.__name__} '
                 f'with replace_maxpools={self.hparams.replace_maxpools}, '
                 f'replace_convs={self.hparams.replace_convs}. '
                 f'Model now has {num_blurpool_layers} BlurMaxPool2d '
                 f'and {num_blurconv_layers} BlurConv2D layers.')

        logger.metric_fit({
            'blurpool/num_blurpool_layers': num_blurpool_layers,
            'blurpool/num_blurconv_layers': num_blurconv_layers,
        })
