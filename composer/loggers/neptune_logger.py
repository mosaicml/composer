# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Neptune.ai <https://neptune.ai/>`_."""

__all__ = ['NeptuneLogger']

import os
import pathlib
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np
import torch

from composer._version import __version__
from composer.loggers import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from composer import Logger
    from composer.core import State


class NeptuneLogger(LoggerDestination):
    """LoggerDestination implementation that sends the logs to Neptune.ai.

    Args:
        project (str, optional): The name of your Neptune project. If nothing is
            specified, environment variable NEPTUNE_PROJECT will be used.
        api_token (str, optional): Your Neptune.ai api key. If nothing is specified,
            environment variable NEPTUNE_API_TOKEN will be used.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            (default: ``True``).
        upload_artifacts (bool, optional): Whether the logger should upload artifacts to Neptune.ai.
            (default: ``False``).
        base_namespace (str, optional): The name of the base namespace to log the metadata to. (default: "training")
        neptune_kwargs (Dict[str, Any], optional): Any additional kwargs to neptune.init_run
            function. (For help see `Run documentation <https://docs.neptune.ai/api/neptune/#init_run>`_)
    """
    METRIC_NAMESPACE = 'metrics'
    HYPERPARAM_NAMESPACE = 'hyperparameters'
    TRACE_NAMESPACE = 'traces'
    INTEGRATION_VERSION_KEY = 'source_code/integrations/neptune-MosaicML'

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        rank_zero_only: bool = True,
        upload_artifacts: bool = False,
        base_namespace: str = 'training',
        **neptune_kwargs,
    ) -> None:
        try:
            from neptune import Run
            from neptune.internal.utils import verify_type
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='neptune',
                                                conda_package='neptune',
                                                conda_channel='conda-forge') from e

        verify_type('project', project, (str, type(None)))
        verify_type('api_token', api_token, (str, type(None)))
        verify_type('rank_zero_only', rank_zero_only, bool)
        verify_type('upload_artifacts', upload_artifacts, bool)
        verify_type('base_namespace', base_namespace, str)

        if not base_namespace:
            raise ValueError("Argument 'base_namespace' cannot be an empty string.")

        self._project = project
        self._api_token = api_token
        self._rank_zero_only = rank_zero_only
        self._upload_artifacts = upload_artifacts
        self._base_namespace = base_namespace
        self._neptune_kwargs = neptune_kwargs

        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0
        self._neptune_run = None
        self._base_handler = None

        if self._enabled:
            self._neptune_run = Run(
                project=self._project,
                api_token=self._api_token,
                **self._neptune_kwargs,
            )
            self._base_handler = self._neptune_run[self._base_namespace]

        super(NeptuneLogger, self).__init__()

    @property
    def neptune_run(self):
        return self._neptune_run

    @property
    def base_handler(self):
        return self._base_handler

    def init(self, state: 'State', logger: 'Logger') -> None:
        del logger  # unused

        tags = [state.run_name]

        if not self._rank_zero_only:
            tags.append(f'rank{dist.get_global_rank()}')

        if self._enabled:
            assert self._neptune_run is not None
            self._neptune_run['sys/tags'].add(tags)
            self._neptune_run[self.INTEGRATION_VERSION_KEY] = __version__

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._enabled:
            return

        assert self._base_handler is not None

        from neptune.utils import stringify_unsupported

        self._base_handler[NeptuneLogger.METRIC_NAMESPACE].append(stringify_unsupported(metrics), step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        assert self._base_handler is not None

        self._base_handler[NeptuneLogger.HYPERPARAM_NAMESPACE] = stringify_unsupported(hyperparameters)

    def log_traces(self, traces: Dict[str, Any]):
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        assert self._base_handler is not None

        self._base_handler[NeptuneLogger.TRACE_NAMESPACE] = stringify_unsupported(traces)

    def can_upload_files(self) -> bool:
        """Whether the logger supports uploading files."""
        return self._enabled and self._upload_artifacts

    def upload_file(
        self,
        state: 'State',
        remote_file_name: str,
        file_path: pathlib.Path,
        *,
        overwrite: bool = False,
    ):
        if not self.can_upload_files():
            return

        assert self._neptune_run is not None

        neptune_path = f'{self._base_namespace}/{remote_file_name}'
        if self._neptune_run.exists(neptune_path) and not overwrite:

            warnings.warn(f"The file '{neptune_path}' already exists and overwrite is set to False."
                          'No action will be taken.')
            return

        del state  # unused

        assert self._base_handler is not None
        self._base_handler[remote_file_name].upload(str(file_path))

    def download_file(
        self,
        remote_file_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        del progress_bar  # not supported

        if not self._enabled:
            return

        if os.path.exists(os.path.join(
                destination,
                remote_file_name,
        )) and not overwrite:
            warnings.warn(f"Destination '{destination}' already exists and overwrite is set to False."
                          'No action will be taken.')
            return

        assert self._neptune_run is not None
        assert self._base_handler is not None

        file_path = f'{self._base_namespace}/{remote_file_name}'
        if not self._neptune_run.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found')

        self._base_handler[remote_file_name].download(destination=destination)

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[Dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[Dict[int, str]] = None,
        use_table: bool = True,
    ):
        if not self._enabled:
            return

        assert self._base_handler is not None

        if not isinstance(images, Sequence) and images.ndim <= 3:
            images = [images]

        for img in images:
            self._base_handler[name].append(img, step=step)

    def post_close(self) -> None:
        if not self._enabled:
            return

        if self._neptune_run:
            self._neptune_run.stop()
