# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List, Tuple, Type

import pytest
from torch.utils.data import DataLoader

import composer.callbacks
import composer.loggers
import composer.profiler
from composer import Callback
from composer.callbacks import (EarlyStopper, ExportForInferenceCallback, FreeOutputs, Generate, ImageVisualizer,
                                MemoryMonitor, MemorySnapshot, MLPerfCallback, OOMObserver, SpeedMonitor,
                                SystemMetricsMonitor, ThresholdStopper)
from composer.loggers import (CometMLLogger, ConsoleLogger, LoggerDestination, MLFlowLogger, NeptuneLogger,
                              ProgressBarLogger, RemoteUploaderDownloader, TensorboardLogger, WandBLogger)
from composer.models.base import ComposerModel
from composer.utils import dist
from composer.utils.device import get_device
from tests.common import get_module_subclasses
from tests.common.datasets import RandomClassificationDataset, dummy_gpt_lm_dataloader
from tests.common.models import SimpleModel, configure_tiny_gpt2_hf_model

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
    import comet_ml
    _COMETML_INSTALLED = True
    os.environ['COMET_API_KEY']
    del comet_ml  # unused
except ImportError:
    _COMETML_INSTALLED = False
# If COMET_API_KEY not set.
except KeyError:
    _COMETML_INSTALLED = False

try:
    import mlperf_logging
    _MLPERF_INSTALLED = True
    del mlperf_logging
except ImportError:
    _MLPERF_INSTALLED = False

try:
    import mlflow
    _MLFLOW_INSTALLED = True
    del mlflow
except ImportError:
    _MLFLOW_INSTALLED = False

try:
    import libcloud
    _LIBCLOUD_INSTALLED = True
    del libcloud  # unused
except ImportError:
    _LIBCLOUD_INSTALLED = False

try:
    import pynmvl
    _PYNMVL_INSTALLED = True
    del pynmvl  # unused
except ImportError:
    _PYNMVL_INSTALLED = False

try:
    import neptune
    _NEPTUNE_INSTALLED = True
    del neptune  # unused
except ImportError:
    _NEPTUNE_INSTALLED = False

_callback_kwargs: Dict[Type[Callback], Dict[str, Any],] = {
    Generate: {
        'prompts': ['a', 'b', 'c'],
        'interval': '1ba',
        'batch_size': 2,
        'max_new_tokens': 20
    },
    RemoteUploaderDownloader: {
        'bucket_uri': 'libcloud://.',
        'backend_kwargs': {
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
        'monitor': 'MulticlassAccuracy',
        'dataloader_label': 'train',
        'threshold': 0.99,
    },
    EarlyStopper: {
        'monitor': 'MulticlassAccuracy',
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
    NeptuneLogger: {
        'neptune_project': 'test_project',
        'neptune_api_token': 'test_token',
        'mode': 'debug',
    },
}

_callback_marks: Dict[Type[Callback], List[pytest.MarkDecorator],] = {
    RemoteUploaderDownloader: [
        pytest.mark.filterwarnings(
            # post_close might not be called if being used outside of the trainer
            r'ignore:Implicitly cleaning up:ResourceWarning'),
        pytest.mark.skipif(not _LIBCLOUD_INSTALLED, reason='Libcloud is optional')
    ],
    MemoryMonitor: [
        pytest.mark.filterwarnings(
            r'ignore:The memory monitor only works on CUDA devices, but the model is on cpu:UserWarning')
    ],
    MemorySnapshot: [
        pytest.mark.filterwarnings(
            r'ignore:The memory snapshot only works on CUDA devices, but the model is on cpu:UserWarning')
    ],
    OOMObserver: [
        pytest.mark.filterwarnings(
            r'ignore:The oom observer only works on CUDA devices, but the model is on cpu:UserWarning')
    ],
    MLPerfCallback: [pytest.mark.skipif(not _MLPERF_INSTALLED, reason='MLPerf is optional')],
    WandBLogger: [
        pytest.mark.filterwarnings(r'ignore:unclosed file:ResourceWarning'),
        pytest.mark.skipif(not _WANDB_INSTALLED, reason='Wandb is optional'),
    ],
    ProgressBarLogger: [
        pytest.mark.filterwarnings(
            r'ignore:Specifying the ProgressBarLogger via `loggers` is not recommended as.*:Warning')
    ],
    ConsoleLogger: [
        pytest.mark.filterwarnings(r'ignore:Specifying the ConsoleLogger via `loggers` is not recommended as.*:Warning')
    ],
    CometMLLogger: [pytest.mark.skipif(not _COMETML_INSTALLED, reason='comet_ml is optional'),],
    TensorboardLogger: [pytest.mark.skipif(not _TENSORBOARD_INSTALLED, reason='Tensorboard is optional'),],
    ImageVisualizer: [pytest.mark.skipif(not _WANDB_INSTALLED, reason='Wandb is optional')],
    MLFlowLogger: [pytest.mark.skipif(not _MLFLOW_INSTALLED, reason='mlflow is optional'),],
    SystemMetricsMonitor: [pytest.mark.skipif(not _PYNMVL_INSTALLED, reason='pynmvl is optional'),],
    NeptuneLogger: [pytest.mark.skipif(not _NEPTUNE_INSTALLED, reason='neptune is optional'),],
}


def get_cb_kwargs(impl: Type[Callback]):
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
    # TODO: (Hanlin) populate this
    implementations = []
    ans = [_to_pytest_param(impl) for impl in implementations]
    return ans


def get_cb_model_and_datasets(cb: Callback,
                              dl_size=100,
                              **default_dl_kwargs) -> Tuple[ComposerModel, DataLoader, DataLoader]:
    if isinstance(cb, Generate):
        if get_device(None).name == 'cpu' and dist.get_world_size() > 1:
            pytest.xfail(
                'GPT2 is not currently supported with DDP. See https://github.com/huggingface/transformers/issues/22482 for more details.'
            )
        return (configure_tiny_gpt2_hf_model(), dummy_gpt_lm_dataloader(size=dl_size),
                dummy_gpt_lm_dataloader(size=dl_size))
    model = SimpleModel()
    if isinstance(cb, FreeOutputs):
        model.get_metrics = lambda is_train=False: {}
    return (model, DataLoader(RandomClassificationDataset(size=dl_size), **default_dl_kwargs),
            DataLoader(RandomClassificationDataset(size=dl_size), **default_dl_kwargs))
