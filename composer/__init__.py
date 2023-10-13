# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Some key classes are available directly in the ``composer`` namespace."""

from composer._version import __version__
from composer.core import Algorithm, Callback, DataSpec, Engine, Evaluator, Event, State, Time, Timestamp, TimeUnit
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.trainer import Trainer

try:
    import transformers
    import flash_attn
    import version
    
    # Before importing any transformers models, we need to disable transformers flash attention if
    # we are in an environment with flash attention version <2. Transformers hard errors on a not properly
    # gated import otherwise.
    if version.parse(flash_attn.__version__) < version.parse('2.0.0'):
        transformers.utils.is_flash_attn_available = lambda : False
except:
    pass

__all__ = [
    'Algorithm',
    'Callback',
    'DataSpec',
    'Engine',
    'Evaluator',
    'Event',
    'State',
    'Time',
    'Timestamp',
    'TimeUnit',
    'Logger',
    'ComposerModel',
    'Trainer',
]
