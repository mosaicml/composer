# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Composer."""

from composer.core import Algorithm, Callback, DataSpec, Engine, Evaluator, Event, State, Time, Timestamp, TimeUnit
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.trainer import Trainer

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

__version__ = '0.8.0'
