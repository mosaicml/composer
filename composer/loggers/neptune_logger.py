# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log training metadata to [neptune.ai](https://neptune.ai/)."""

__all__ = ['NeptuneLogger']

import os
import pathlib
import warnings
from functools import partial
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import numpy as np
import torch
from packaging.version import Version

from composer._version import __version__
from composer.loggers import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from composer import Logger
    from composer.core import State

NEPTUNE_MODE_TYPE = Literal['async', 'sync', 'offline', 'read-only', 'debug']
NEPTUNE_VERSION_WITH_PROGRESS_BAR = Version('1.9.0')


class NeptuneLogger(LoggerDestination):
    """Log to `neptune.ai <https://neptune.ai/>`_.

    For instructions, see the
    `integration guide <https://docs.neptune.ai/integrations/mosaicml_composer/>`_.

    Args:
        project (str, optional): The name of your Neptune project,
            in the form "workspace-name/project-name". If you leave it empty, the
            ``NEPTUNE_PROJECT`` environment variable will be used.
        api_token (str, optional): Your Neptune API token.
            You can leave out this argument if you save your token to the
            ``NEPTUNE_API_TOKEN`` environment variable (recommended).
            You can find your API token in the user menu of the Neptune web app.
        rank_zero_only (bool): Whether to log only on the rank-zero process (default: ``True``).
        upload_checkpoints (bool): Whether the logger should upload checkpoints to Neptune
            (default: ``False``).
        base_namespace (str, optional): The name of the base namespace where the metadata
            is logged (default: "training").
        neptune_kwargs (dict[str, Any], optional): Any additional keyword arguments to the
            ``neptune.init_run()`` function. For options, see the
            `Run API reference <https://docs.neptune.ai/api/neptune/#init_run>`_.
    """
    metric_namespace = 'metrics'
    hyperparam_namespace = 'hyperparameters'
    trace_namespace = 'traces'
    integration_version_key = 'source_code/integrations/neptune-MosaicML'

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        api_token: Optional[str] = None,
        rank_zero_only: bool = True,
        upload_checkpoints: bool = False,
        base_namespace: str = 'training',
        mode: Optional[NEPTUNE_MODE_TYPE] = None,
        **neptune_kwargs,
    ) -> None:
        try:
            from neptune.internal.utils import verify_type
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='neptune',
                conda_package='neptune',
                conda_channel='conda-forge',
            ) from e

        verify_type('project', project, (str, type(None)))
        verify_type('api_token', api_token, (str, type(None)))
        verify_type('rank_zero_only', rank_zero_only, bool)
        verify_type('upload_checkpoints', upload_checkpoints, bool)
        verify_type('base_namespace', base_namespace, str)

        if not base_namespace:
            raise ValueError("Argument 'base_namespace' cannot be an empty string.")

        self._project = project
        self._api_token = api_token
        self._rank_zero_only = rank_zero_only

        self._upload_checkpoints = upload_checkpoints

        self._base_namespace = base_namespace
        self._neptune_kwargs = neptune_kwargs

        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        self._mode: Optional[NEPTUNE_MODE_TYPE] = mode if self._enabled else 'debug'

        self._neptune_run = None
        self._base_handler = None

        self._metrics_dict: dict[str, int] = {}  # used to prevent duplicate step logging

        super().__init__()

    @property
    def neptune_run(self):
        """Gets the Neptune run object from a NeptuneLogger instance.

        To log additional metadata to the run, access a path inside the run and assign metadata
        with ``=`` or other `Neptune logging methods <https://docs.neptune.ai/logging/methods/>`_.
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

        Use the handler to log extra metadata to the run and organize it under the base namespace
        (default: "training"). You can operate on it like a run object: Access a path inside the
        handler and assign metadata to it with ``=`` or other
        `Neptune logging methods <https://docs.neptune.ai/logging/methods/>`_.
        """
        return self.neptune_run[self._base_namespace]

    def init(self, state: 'State', logger: 'Logger') -> None:
        del logger  # unused

        self.base_handler['rank'] = dist.get_global_rank()

        if self._enabled:
            self.neptune_run['sys/name'] = state.run_name
            self.neptune_run[self.integration_version_key] = __version__

    def _sanitize_metrics(self, metrics: dict[str, float], step: Optional[int]) -> dict[str, float]:
        """Sanitize metrics to prevent duplicate step logging.

        Args:
            metrics (dict[str, float]): Metrics to log.
            step (Optional[int]): Step to log metrics at.

        Returns:
            dict[str, float]: Sanitized metrics.
        """
        keys_to_delete: set[str] = set()

        for k in metrics:
            self._process_single_metric(k, step, keys_to_delete)

        return dict(filter(lambda x: x[0] not in keys_to_delete, metrics.items()))

    def _process_single_metric(self, metric_key: str, step: Optional[int], keys_to_delete: set[str]) -> None:
        if metric_key not in self._metrics_dict:
            self._metrics_dict[metric_key] = step if step is not None else 0
        else:
            if step is not None:
                if step <= self._metrics_dict[metric_key]:
                    # we cannot insert metrics earlier than or in place of an existing metric point
                    keys_to_delete.add(metric_key)
                else:
                    self._metrics_dict[metric_key] = step
            else:
                self._metrics_dict[metric_key] += 1

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        if metrics_to_log := self._sanitize_metrics(metrics, step):
            self.base_handler[NeptuneLogger.metric_namespace].append(stringify_unsupported(metrics_to_log), step=step)

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        self.base_handler[NeptuneLogger.hyperparam_namespace] = stringify_unsupported(hyperparameters)

    def log_traces(self, traces: dict[str, Any]):
        if not self._enabled:
            return

        from neptune.utils import stringify_unsupported

        self.base_handler[NeptuneLogger.trace_namespace] = stringify_unsupported(traces)

    def can_upload_files(self) -> bool:
        """Whether the logger supports uploading files."""
        return self._enabled and self._upload_checkpoints

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

        if file_path.is_symlink() or file_path.suffix.lower() == '.symlink':
            return  # skip symlinks

        neptune_path = f'{self._base_namespace}/{remote_file_name}'
        if self.neptune_run.exists(neptune_path) and not overwrite:

            warnings.warn(
                f"The file '{neptune_path}' already exists and overwrite is set to False."
                'No action will be taken.',
            )
            return

        del state  # unused

        from neptune.types import File

        with open(str(file_path), 'rb') as fp:
            self.base_handler[remote_file_name] = File.from_stream(fp, extension=file_path.suffix)

    def download_file(
        self,
        remote_file_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):

        if not self._enabled:
            return

        if os.path.exists(
            os.path.join(
                destination,
                remote_file_name,
            ),
        ) and not overwrite:
            warnings.warn(
                f"Destination '{destination}' already exists and overwrite is set to False."
                'No action will be taken.',
            )
            return

        file_path = f'{self._base_namespace}/{remote_file_name}'
        if not self.neptune_run.exists(file_path):
            raise FileNotFoundError(f'File {file_path} not found')

        if _is_progress_bar_enabled():
            self.base_handler[remote_file_name].download(destination=destination, progress_bar=progress_bar)
        else:
            del progress_bar
            self.base_handler[remote_file_name].download(destination=destination)

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[dict[int, str]] = None,
        use_table: bool = True,
    ):
        if not self._enabled:
            return

        from neptune.types import File

        if not isinstance(images, Sequence) and images.ndim <= 3:
            images = _validate_image(images, channels_last=channels_last)
            self.base_handler[name].append(File.as_image(images), step=step)

        else:
            images = list(map(partial(_validate_image, channels_last=channels_last), images))
            self.base_handler[name].extend([File.as_image(img) for img in images])

    def post_close(self) -> None:
        if not self._enabled:
            return

        if self._neptune_run:
            self._neptune_run.stop()
            self._neptune_run = None


def _validate_image(img: Union[np.ndarray, torch.Tensor], channels_last: bool) -> np.ndarray:
    img_numpy = img.data.cpu().numpy() if isinstance(img, torch.Tensor) else img

    assert isinstance(img_numpy, np.ndarray)

    # Error out for empty arrays or weird arrays of dimension 0.
    if np.any(np.equal(img_numpy.shape, 0)):
        raise ValueError(f'Got an image (shape {img_numpy.shape}) with at least one dimension being 0! ')

    if not channels_last:
        img_numpy = np.moveaxis(img_numpy, 0, -1)

    return _validate_image_value_range(img_numpy)


def _validate_image_value_range(img: np.ndarray) -> np.ndarray:
    array_min = img.min()
    array_max = img.max()

    if (array_min >= 0 and 1 < array_max <= 255) or (array_min >= 0 and array_max <= 1):
        return img

    from neptune.common.warnings import NeptuneWarning, warn_once

    warn_once(
        'Image value range is not in the expected range of [0.0, 1.0] or [0, 255]. '
        'This might be due to the presence of `transforms.Normalize` in the data pipeline. '
        'Logged images may not display correctly in Neptune.',
        exception=NeptuneWarning,
    )

    return _scale_image_to_0_255(img, array_min, array_max)


def _scale_image_to_0_255(img: np.ndarray, array_min: Union[int, float], array_max: Union[int, float]) -> np.ndarray:
    scaled_image = 255 * (img - array_min) / (array_max - array_min)
    return scaled_image.astype(np.uint8)


def _is_progress_bar_enabled() -> bool:
    return Version(version('neptune')) >= NEPTUNE_VERSION_WITH_PROGRESS_BAR
