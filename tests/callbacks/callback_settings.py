# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Type, Union

import pytest
import yahp as hp

import composer.callbacks
import composer.loggers
import composer.profiler
from composer import Callback
from composer.callbacks import EarlyStopper, ImageVisualizer, MemoryMonitor, SpeedMonitor, ThresholdStopper
from composer.callbacks.callback_hparams_registry import callback_registry
from composer.callbacks.export_for_inference import ExportForInferenceCallback
from composer.callbacks.mlperf import MLPerfCallback
from composer.loggers import ObjectStoreLogger, TensorboardLogger, WandBLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger_hparams_registry import ObjectStoreLoggerHparams, logger_registry
from composer.loggers.progress_bar_logger import ProgressBarLogger
from composer.utils.object_store.libcloud_object_store import LibcloudObjectStore
from tests.common import get_module_subclasses

try:
    import wandb
    _WANDB_INSTALLED = True
    del wandb  # unused
except ImportError:
    _WANDB_INSTALLED = False

try:
    import tensorboard
    _TENSORBOARD_INSTALLED = True
    del tensorboard  # unused
except ImportError:
    _TENSORBOARD_INSTALLED = False

try:
    import mlperf_logging
    _MLPERF_INSTALLED = True
    del mlperf_logging
except ImportError:
    _MLPERF_INSTALLED = False

try:
    import libcloud
    _LIBCLOUD_INSTALLED = True
    del libcloud  # unused
except ImportError:
    _LIBCLOUD_INSTALLED = False

_callback_kwargs: Dict[Union[Type[Callback], Type[hp.Hparams]], Dict[str, Any],] = {
    ObjectStoreLogger: {
        'object_store_cls': LibcloudObjectStore,
        'object_store_kwargs': {
            'provider': 'local',
            'container': '.',
            'provider_kwargs': {
                'key': '.',
            },
        },
        'use_procs': False,
        'num_concurrent_uploads': 1,
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
    ExportForInferenceCallback: {
        'save_format': 'torchscript',
        'save_path': '/tmp/model.pth',
    },
    MLPerfCallback: {
        'root_folder': '.',
        'index': 0,
    },
    SpeedMonitor: {
        'window_size': 1,
    },
    ObjectStoreLoggerHparams: {
        'object_store_hparams': {
            'libcloud': {
                'provider': 'local',
                'container': '.',
                'key_environ': 'KEY_ENVIRON',
            },
        },
        'use_procs': False,
        'num_concurrent_uploads': 1,
    },
}

_callback_marks: Dict[Union[Type[Callback], Type[hp.Hparams]], List[pytest.MarkDecorator],] = {
    ObjectStoreLogger: [
        pytest.mark.filterwarnings(
            # post_close might not be called if being used outside of the trainer
            r'ignore:Implicitly cleaning up:ResourceWarning'),
        pytest.mark.skipif(not _LIBCLOUD_INSTALLED, reason='Libcloud is optional')
    ],
    MemoryMonitor: [
        pytest.mark.filterwarnings(
            r'ignore:The memory monitor only works on CUDA devices, but the model is on cpu:UserWarning')
    ],
    MLPerfCallback: [pytest.mark.skipif(not _MLPERF_INSTALLED, reason='MLPerf is optional')],
    WandBLogger: [
        pytest.mark.filterwarnings(r'ignore:unclosed file:ResourceWarning'),
        pytest.mark.skipif(not _WANDB_INSTALLED, reason='Wandb is optional'),
    ],
    ProgressBarLogger: [
        pytest.mark.filterwarnings(
            r'ignore:Specifying the ProgressBarLogger via `loggers` is deprecated:DeprecationWarning')
    ],
    TensorboardLogger: [pytest.mark.skipif(not _TENSORBOARD_INSTALLED, reason='Tensorboard is optional'),],
    ObjectStoreLoggerHparams: [
        pytest.mark.filterwarnings(
            # post_close might not be called if being used outside of the trainer
            r'ignore:Implicitly cleaning up:ResourceWarning',),
    ],
    ImageVisualizer: [pytest.mark.skipif(not _WANDB_INSTALLED, reason='Wandb is optional')],
}


def get_cb_kwargs(impl: Union[Type[Callback], Type[hp.Hparams]]):
    return _callback_kwargs.get(impl, {})


def _to_pytest_param(impl):
    if impl not in _callback_marks:
        return pytest.param(impl)
    else:
        marks = _callback_marks[impl]
        return pytest.param(impl, marks=marks)


def get_cbs_and_marks(callbacks: bool = False, loggers: bool = False, profilers: bool = False):
    """Returns a list of :class:`pytest.mark.param` objects for all :class:`.Callback`.
    The callbacks are correctly annotated with ``skipif`` marks for optional dependencies
    and ``filterwarning`` marks for any warnings that might be emitted and are safe to ignore

    This function is meant to be used like this::

        import pytest
        from tests.callbacks.callback_settings import get_cbs_and_marks, get_cb_kwargs

        @pytest.mark.parametrize("cb_cls",get_cbs_and_marks(callbacks=True, loggers=True, profilers=True))
        def test_something(cb_cls: Type[Callback]):
            cb_kwargs = get_cb_kwargs(cb_cls)
            cb = cb_cls(**cb_kwargs)
            assert isinstance(cb, Callback)
    """
    implementations = []
    if callbacks:
        implementations.extend(get_module_subclasses(composer.callbacks, Callback))
    if loggers:
        implementations.extend(get_module_subclasses(composer.loggers, LoggerDestination))
    if profilers:
        implementations.extend(get_module_subclasses(composer.profiler, Callback))
    ans = [_to_pytest_param(impl) for impl in implementations]
    if not len(ans):
        raise ValueError('callbacks, loggers, or profilers must be True')
    return ans


def get_cb_hparams_and_marks():
    """Returns a list of :class:`pytest.mark.param` objects for all ``callback_registry``
    and ``logger_registry``entries.

    The callbacks are correctly annotated with ``skipif`` marks for optional dependencies
    and ``filterwarning`` marks for any warnings that might be emitted and are safe to ignore

    This function is meant to be used like this::

        import pytest
        from tests.common.hparams import construct_from_yaml
        from tests.callbacks.callback_settings import get_cb_hparams_and_marks, get_cb_kwargs

        @pytest.mark.parametrize("constructor",get_cb_hparams_and_marks())
        def test_something(constructor: Callable, yaml_dict: Dict[str, Any]):
            yaml_dict = get_cb_kwargs(constructor)
            construct_from_yaml(constructor, yaml_dict=yaml_dict)
    """
    implementations = [
        *callback_registry.values(),
        *logger_registry.values(),
    ]
    ans = [_to_pytest_param(impl) for impl in implementations]
    return ans
