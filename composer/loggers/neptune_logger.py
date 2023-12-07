# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2023 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log training metadata to [neptune.ai](https://neptune.ai/)."""

__all__ = ['NeptuneLogger']

import os
import pathlib
import textwrap
import warnings
from functools import partial
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
    """LoggerDestination implementation that sends the logs to Neptune.

    Args:
        project (str, optional): The name of your Neptune project, in the form `workspace-name/project-name`.
            If you leave it empty, the NEPTUNE_PROJECT environment variable will be used.
        api_token (str, optional): Your Neptune API token. You can leave out this argument if you save your token to the
            NEPTUNE_API_TOKEN environment variable (recommended).
            You can find your API token in the user menu of the Neptune web app.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process. Default: ``True``.
        upload_artifacts (bool, optional): Whether the logger should upload artifacts to Neptune. Default: ``False``.
        base_namespace (str, optional): The name of the base namespace to log the metadata to. Default: "training".
        neptune_kwargs (Dict[str, Any], optional): Any additional keyword arguments to the neptune.init_run()
            function. For options, see the [Run API reference](https://docs.neptune.ai/api/neptune/#init_run) in the
            Neptune docs.

    Example:
        Default Neptune logger:
        >>> from composer.loggers import NeptuneLogger
        >>> neptune_logger = NeptuneLogger()
        >>> trainer = Trainer(loggers=neptune_logger, ...)
        >>> trainer.fit()

        More options:
        >>> neptune_logger = NeptuneLogger(
        ...     project="ml-team/classification",
        ...     upload_artifacts=True,
        ...     dependencies="infer",  # Neptune Run kwarg
        )
        >>> trainer = Trainer(loggers=neptune_logger, ...)
        >>> trainer.fit()

    For more, see the [Neptune-Composer integration guide](https://docs.neptune.ai/integrations/composer/).
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

        mode = self._neptune_kwargs.pop('mode', 'async')

        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self._mode = mode if self._enabled else 'debug'

        self._neptune_run = None
        self._base_handler = None

        super(NeptuneLogger, self).__init__()

    @property
    def neptune_run(self):
        """Gets the Neptune run object from a NeptuneLogger instance.

        You can log additional metadata to the run by accessing a path inside the run and assigning metadata to it
        with "=" or [Neptune logging methods](https://docs.neptune.ai/logging/methods/).

        Example:
            >>> neptune_logger = NeptuneLogger()
            >>> trainer = Trainer(loggers=neptune_logger, ...)
            >>> trainer.fit()
            >>> neptune_logger.neptune_run["some_metric"] = 1
            >>> trainer.close()
        """
        from neptune import Run

        if not self._neptune_run:
            self._neptune_run = Run(
                project=self._project,
                api_token=self._api_token,
                mode=self._mode,
                **self._neptune_kwargs,
            )
        return self._neptune_run

    @property
    def base_handler(self):
        """Gets a handler for the base logging namespace.
    
        Use the handler to log extra metadata to the run and organize it under the base namespace (default: "training").
        You can operate on it like a run object: Access a path inside the handler and assign metadata to it with "=" or
        other [Neptune logging methods](https://docs.neptune.ai/logging/methods/).
    
        Example:
            >>> neptune_logger = NeptuneLogger()
            >>> trainer = Trainer(loggers=neptune_logger, ...)
            >>> trainer.fit()
            >>> neptune_logger.base_handler["some_metric"] = 1
            >>> trainer.close()
            Result: The value `1` is organized under "training/some_metric" inside the run.
        """
        return self.neptune_run[self._base_namespace]

    def init(self, state: 'State', logger: 'Logger') -> None:
        del logger  # unused

        if not self._rank_zero_only:
            self.base_handler['rank'] = dist.get_global_rank()

        if self._enabled:
            self.neptune_run['sys/name'] = state.run_name
            self.neptune_run[self.INTEGRATION_VERSION_KEY] = __version__

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        self.base_handler[NeptuneLogger.METRIC_NAMESPACE].append(stringify_unsupported(metrics), step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        self.base_handler[NeptuneLogger.HYPERPARAM_NAMESPACE] = stringify_unsupported(hyperparameters)

    def log_traces(self, traces: Dict[str, Any]):
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        self.base_handler[NeptuneLogger.TRACE_NAMESPACE] = stringify_unsupported(traces)

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

        neptune_path = f'{self._base_namespace}/{remote_file_name}'
        if self.neptune_run.exists(neptune_path) and not overwrite:

            warnings.warn(f"The file '{neptune_path}' already exists and overwrite is set to False."
                          'No action will be taken.')
            return

        del state  # unused
        self.base_handler[remote_file_name].upload(str(file_path))

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

        file_path = f'{self._base_namespace}/{remote_file_name}'
        if not self.neptune_run.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found')

        self.base_handler[remote_file_name].download(destination=destination)

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

        unused_args = (masks, mask_class_labels, use_table)  # Unused
        if any(unused_args):
            warnings.warn(
                textwrap.dedent(f"""NeptuneLogger does not support masks, class labels, or tables of images,
                                  but got {masks=}, {mask_class_labels=}, {use_table=}"""))

        if not isinstance(images, Sequence) and images.ndim <= 3:
            images = [images]

        images = list(map(partial(_validate_image, channels_last=channels_last), images),)

        self.base_handler[name].extend(images, step=step)

    def post_close(self) -> None:
        if not self._enabled:
            return

        if self._neptune_run:
            self._neptune_run.stop()
            self._neptune_run = None


def _validate_image(img: Union[np.ndarray, torch.Tensor], channels_last: bool) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()

    if not isinstance(img, np.ndarray):
        raise ValueError(f'Image should be of type np.ndarray or torch.Tensor. Got {type(img)}')

    # Error out for empty arrays or weird arrays of dimension 0.
    if np.any(np.equal(img.shape, 0)):
        raise ValueError(f'Got an image (shape {img.shape}) with at least one dimension being 0! ')

    if not channels_last:
        img = np.moveaxis(img, 0, -1)

    return img
