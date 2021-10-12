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
    """Applies BlurPool algorithm to the provided model. Performs an in-place
    replacement of eligible convolution and pooling layers.

    Args:
        model (torch.nn.Module): model to transform
        replace_convs (bool): replace eligible Conv2D with BlurConv2d. Default: True.
        replace_maxpools (bool): replace eligible MaxPool2d with BlurMaxPool2d. Default: True.
        blur_first (bool): for replace_convs, blur input before conv. Default: True
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

    replace_convs: bool = hp.optional('Replace Conv2d with BlurConv2d if stride > 1', default=True)
    replace_maxpools: bool = hp.optional('Replace MaxPool2d with BlurMaxPool2d', default=True)
    blur_first: bool = hp.optional('Blur input before Conv', default=True)

    def initialize_object(self) -> "BlurPool":
        return BlurPool(**asdict(self))


def _maybe_replace_strided_conv2d(module: torch.nn.Conv2d, module_index: int, blur_first: bool):
    if (np.max(module.stride) > 1 and module.in_channels >= 16):
        return BlurConv2d.from_conv2d(module, module_index, blur_first=blur_first)
    return None


class BlurPool(Algorithm):
    """Algorithm to apply BlurPool to the model. Runs on Event.INIT. This algorithm should
    be applied before the model has been moved to devices.

    Args:
        replace_convs (bool): replace eligible Conv2D with BlurConv2d. Default: True.
        replace_maxpools (bool): replace eligible MaxPool2d with BlurMaxPool2d. Default: True.
        blur_first (bool): for replace_convs, blur input before conv. Default: True
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
        """ Runs on Event.INIT
        """
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        """ Applies BlurPool
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
        # TODO: standardize algorithm logging
        log.info(f'Applied BlurPool to model {state.model.__class__.__name__} '
                 f'with replace_maxpools={self.hparams.replace_maxpools}, '
                 f'replace_convs={self.hparams.replace_convs}. '
                 f'Model now has {num_blurpool_layers} BlurMaxPool2d '
                 f'and {num_blurconv_layers} BlurConv2D layers.')

        logger.metric_fit({
            'blurpool/num_blurpool_layers': num_blurpool_layers,
            'blurpool/num_blurconv_layers': num_blurconv_layers,
        })
