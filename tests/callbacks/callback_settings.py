# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Type, Union

import pytest

import composer.callbacks
import composer.loggers
import composer.profiler
from composer import Callback
from composer.callbacks import EarlyStopper, MemoryMonitor, MLPerfCallback, ThresholdStopper
from composer.callbacks.callback_hparams import (CallbackHparams, EarlyStopperHparams, MLPerfCallbackHparams,
                                                 ThresholdStopperHparams, callback_registry)
from composer.loggers import LoggerDestination, ObjectStoreLogger, ProgressBarLogger, WandBLogger
from composer.loggers.logger_hparams import LoggerDestinationHparams, ObjectStoreLoggerHparams, logger_registry
from tests.common import get_module_subclasses

try:
    import wandb
    _WANDB_INSTALLED = True
    del wandb  # unused
except ImportError:
    _WANDB_INSTALLED = False

try:
    import mlperf_logging
    _MLPERF_INSTALLED = True
    del mlperf_logging
except ImportError:
    _MLPERF_INSTALLED = False

_callback_kwargs: Dict[Union[Type[Callback], Type[CallbackHparams], Type[LoggerDestinationHparams]],
                       Dict[str, Any],] = {
                           ObjectStoreLogger: {
                               'use_procs': False,
                               'num_concurrent_uploads': 1,
                               'provider': 'local',
                               'container': '.',
                               'provider_kwargs': {
                                   'key': '.',
                               },
                           },
                           ThresholdStopper: {
                               'monitor': 'Accuracy',
                               'dataloader_label': 'train',
                               'threshold': 0.99,
                           },
                           EarlyStopper: {
                               'monitor': 'Accuracy',
                               'dataloader_label': 'train',
                           },
                           MLPerfCallback: {
                               'root_folder': '.',
                               'index': 0,
                           },
                           ObjectStoreLoggerHparams: {
                               'object_store_hparams': {
                                   'provider': 'local',
                                   'container': '.',
                                   'key_environ': 'KEY_ENVIRON',
                               },
                               'use_procs': False,
                               'num_concurrent_uploads': 1,
                           },
                       }

_callback_marks: Dict[Union[
    Type[Callback], Type[CallbackHparams], Type[LoggerDestinationHparams]], List[pytest.MarkDecorator],] = {
        ObjectStoreLogger: [
            pytest.mark.filterwarnings(
                # post_close might not be called if being used outside of the trainer
                r'ignore:Implicitly cleaning up:ResourceWarning')
        ],
        MemoryMonitor: [
            pytest.mark.filterwarnings(
                r'ignore:The memory monitor only works on CUDA devices, but the model is on cpu:UserWarning')
        ],
        MLPerfCallback: [pytest.mark.skipif(not _MLPERF_INSTALLED, reason="MLPerf is optional")],
        WandBLogger: [
            pytest.mark.filterwarnings(r'ignore:unclosed file:ResourceWarning'),
            pytest.mark.skipif(not _WANDB_INSTALLED, reason="Wandb is optional"),
        ],
        ProgressBarLogger: [
            pytest.mark.filterwarnings(
                r"ignore:Specifying the ProgressBarLogger via `loggers` is deprecated:DeprecationWarning")
        ],
        ObjectStoreLoggerHparams: [
            pytest.mark.filterwarnings(
                # post_close might not be called if being used outside of the trainer
                r'ignore:Implicitly cleaning up:ResourceWarning',),
        ],
    }


def get_cb_kwargs(impl: Union[Type[Callback], Type[CallbackHparams], Type[LoggerDestinationHparams]]):
    return _callback_kwargs.get(impl, {})


# Manually inject some of the hparam classes into the settings.
# TODO(ravi): Remove this with autoyahp
_callback_kwargs[ThresholdStopperHparams] = _callback_kwargs[ThresholdStopper]
_callback_kwargs[EarlyStopperHparams] = _callback_kwargs[EarlyStopper]
_callback_kwargs[MLPerfCallbackHparams] = _callback_kwargs[MLPerfCallback]
_callback_marks[MLPerfCallbackHparams] = _callback_marks[MLPerfCallback]


def _to_pytest_param(impl):
    if impl not in _callback_marks:
        return pytest.param(impl)
    else:
        marks = _callback_marks[impl]
        return pytest.param(impl, marks=marks)


def get_cbs_and_marks():
    """Returns a list of :class:`pytest.mark.param` objects for all :class:`.Callback`.
    The callbacks are correctly annotated with ``skipif`` marks for optional dependencies
    and ``filterwarning`` marks for any warnings that might be emitted and are safe to ignore

    This function is meant to be used like this::

        import pytest
        from tests.callbacks.callback_settings import get_cbs_and_marks, get_cb_kwargs

        @pytest.mark.parametrize("cb_cls",get_cbs_and_marks())
        def test_something(cb_cls: Type[Callback]):
            cb_kwargs = get_cb_kwargs(cb_cls)
            cb = cb_cls(**cb_kwargs)
            assert isinstance(cb, Callback)
    """
    implementations = [
        *get_module_subclasses(composer.callbacks, Callback),
        *get_module_subclasses(composer.loggers, LoggerDestination),
        *get_module_subclasses(composer.profiler, Callback),
    ]
    ans = [_to_pytest_param(impl) for impl in implementations]
    return ans


def get_cb_hparams_and_marks():
    """Returns a list of :class:`pytest.mark.param` objects for all ``callback_registry``
    and ``logger_registry``entries.

    The callbacks are correctly annotated with ``skipif`` marks for optional dependencies
    and ``filterwarning`` marks for any warnings that might be emitted and are safe to ignore

    This function is meant to be used like this::

        import pytest
        from tests.common.hparams import assert_yaml_loads
        from tests.callbacks.callback_settings import get_cb_hparams_and_marks, get_cb_kwargs

        @pytest.mark.parametrize("constructor",get_cb_hparams_and_marks())
        def test_something(constructor: Callable, yaml_dict: Dict[str, Any]):
            yaml_dict = get_cb_kwargs(constructor)
            assert_yaml_loads(constructor, yaml_dict=yaml_dict)
    """
    implementations = [
        *callback_registry.values(),
        *logger_registry.values(),
    ]
    ans = [_to_pytest_param(impl) for impl in implementations]
    return ans
