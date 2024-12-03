# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Weights and Biases <https://wandb.ai/>`_."""

from __future__ import annotations

import atexit
import copy
import os
import pathlib
import re
import sys
import tempfile
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import torch

from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import MissingConditionalImportError, dist

if TYPE_CHECKING:
    from composer.core import State

__all__ = ['WandBLogger']


class WandBLogger(LoggerDestination):
    """Log to `Weights and Biases <https://wandb.ai/>`_.

    Args:
        project (str, optional): WandB project name.
        group (str, optional): WandB group name.
        name (str, optional): WandB run name.
            If not specified, the :attr:`.State.run_name` will be used.
        entity (str, optional): WandB entity name.
        tags (list[str], optional): WandB tags.
        log_artifacts (bool, optional): Whether to log
            `artifacts <https://docs.wandb.ai/ref/python/artifact>`_ (Default: ``False``).
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            When logging `artifacts <https://docs.wandb.ai/ref/python/artifact>`_, it is
            highly recommended to log on all ranks.  Artifacts from ranks ≥1 will not be
            stored, which may discard pertinent information (default: ``True``).
        init_kwargs (dict[str, Any], optional): Any additional init kwargs
            ``wandb.init`` (see
            `WandB documentation <https://docs.wandb.ai/ref/python/init>`_).
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list[str]] = None,
        log_artifacts: bool = False,
        rank_zero_only: bool = True,
        init_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            import wandb
        except ImportError as e:
            raise MissingConditionalImportError(
                extra_deps_group='wandb',
                conda_package='wandb',
                conda_channel='conda-forge',
            ) from e

        del wandb  # unused
        if log_artifacts and rank_zero_only and dist.get_world_size() > 1:
            warnings.warn((
                'When logging artifacts, `rank_zero_only` should be set to False. '
                'Artifacts from other ranks will not be collected, leading to a loss of information required to '
                'restore from checkpoints.'
            ))
        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        if init_kwargs is None:
            init_kwargs = {}

        if project is not None:
            init_kwargs['project'] = project

        if group is not None:
            init_kwargs['group'] = group

        if name is not None:
            init_kwargs['name'] = name

        if entity is not None:
            init_kwargs['entity'] = entity

        if tags is not None:
            init_kwargs['tags'] = tags

        self._rank_zero_only = rank_zero_only
        self._log_artifacts = log_artifacts
        self._init_kwargs = init_kwargs
        self._is_in_atexit = False

        # Set these variable directly to allow fetching an Artifact **without** initializing a WandB run
        # When used as a LoggerDestination, these values are overriden from global rank 0 to all ranks on Event.INIT
        self.entity = entity
        self.project = project

        self.run_dir: Optional[str] = None
        self.run_url: Optional[str] = None

        self.table_dict = {}

    def _set_is_in_atexit(self):
        self._is_in_atexit = True

    def log_hyperparameters(self, hyperparameters: dict[str, Any]):
        if self._enabled:
            import wandb
            wandb.config.update(hyperparameters)

    def log_table(
        self,
        columns: list[str],
        rows: list[list[Any]],
        name: str = 'Table',
        step: Optional[int] = None,
    ) -> None:
        if self._enabled:
            import wandb
            table = wandb.Table(columns=columns, rows=rows)
            wandb.log({name: table}, step=step)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            import wandb

            # wandb.log alters the metrics dictionary object, so we deepcopy to avoid
            # side effects.
            metrics_copy = copy.deepcopy(metrics)
            wandb.log(metrics_copy, step)

    def log_images(
        self,
        images: Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]],
        name: str = 'Images',
        channels_last: bool = False,
        step: Optional[int] = None,
        masks: Optional[dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]]] = None,
        mask_class_labels: Optional[dict[int, str]] = None,
        use_table: bool = False,
    ):
        if self._enabled:
            import wandb
            if not isinstance(images, Sequence) and images.ndim <= 3:
                images = [images]

            # _convert_to_wandb_image doesn't include wrapping with wandb.Image to future
            # proof for when we support masks.
            images_generator = (_convert_to_wandb_image(image, channels_last) for image in images)

            if masks is not None:
                # Create a generator that yields masks in the format wandb wants.
                wandb_masks_generator = _create_wandb_masks_generator(
                    masks,
                    mask_class_labels,
                    channels_last=channels_last,
                )
                wandb_images = (
                    wandb.Image(im, masks=mask_dict) for im, mask_dict in zip(images_generator, wandb_masks_generator)
                )

            else:
                wandb_images = (wandb.Image(image) for image in images_generator)

            if use_table:
                table = wandb.Table(columns=[name])
                for wandb_image in wandb_images:
                    table.add_data(wandb_image)
                wandb.log({name + ' Table': table}, step=step)
            else:
                wandb.log({name: list(wandb_images)}, step=step)

    def init(self, state: State, logger: Logger) -> None:
        import wandb
        del logger  # unused

        # Use the state run name if the name is not set.
        if 'name' not in self._init_kwargs or self._init_kwargs['name'] is None:
            self._init_kwargs['name'] = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            name = self._init_kwargs['name']
            self._init_kwargs['name'] += f'-rank{dist.get_global_rank()}'
            self._init_kwargs['group'] = self._init_kwargs['group'] if 'group' in self._init_kwargs else name
        if self._enabled:
            wandb.init(**self._init_kwargs)
            assert wandb.run is not None, 'The wandb run is set after init'
            if hasattr(wandb.run, 'entity') and hasattr(wandb.run, 'project'):
                entity_and_project = [str(wandb.run.entity), str(wandb.run.project)]
            else:
                # Run does not have attribtues if wandb is in disabled mode, so we must mock it
                entity_and_project = ['disabled', 'disabled']
            self.run_dir = wandb.run.dir
            self.run_url = wandb.run.get_url()
            atexit.register(self._set_is_in_atexit)
        else:
            entity_and_project = [None, None]
        # Share the entity and project across all ranks, so they are available on ranks that did not initialize wandb
        dist.broadcast_object_list(entity_and_project)
        self.entity, self.project = entity_and_project
        assert self.entity is not None, 'entity should be defined'
        assert self.project is not None, 'project should be defined'

    def upload_file(self, state: State, remote_file_name: str, file_path: pathlib.Path, *, overwrite: bool):
        del overwrite  # unused

        if self._enabled and self._log_artifacts:
            import wandb

            # Some WandB-specific alias extraction
            timestamp = state.timestamp
            aliases = ['latest', f'ep{int(timestamp.epoch)}-ba{int(timestamp.batch)}']

            # replace all unsupported characters with periods
            # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
            new_remote_file_name = re.sub(r'[^a-zA-Z0-9-_\.]', '.', remote_file_name)
            if new_remote_file_name != remote_file_name:
                warnings.warn((
                    'WandB permits only alpha-numeric, periods, hyphens, and underscores in file names. '
                    f"The file with name '{remote_file_name}' will be stored as '{new_remote_file_name}'."
                ))

            extension = new_remote_file_name.split('.')[-1]

            metadata = {f'timestamp/{k}': v for (k, v) in state.timestamp.state_dict().items()}
            # if evaluating, also log the evaluation timestamp
            if state.dataloader is not state.train_dataloader:
                # TODO If not actively training, then it is impossible to tell from the state whether
                # the trainer is evaluating or predicting. Assuming evaluation in this case.
                metadata.update({f'eval_timestamp/{k}': v for (k, v) in state.eval_timestamp.state_dict().items()})

            # Change the extension so the checkpoint is compatible with W&B's model registry
            if extension == 'pt':
                extension = 'model'

            wandb_artifact = wandb.Artifact(
                name=new_remote_file_name,
                type=extension,
                metadata=metadata,
            )
            wandb_artifact.add_file(os.path.abspath(file_path))
            wandb.log_artifact(wandb_artifact, aliases=aliases)

    def can_upload_files(self) -> bool:
        """Whether the logger supports uploading files."""
        return True

    def download_file(
        self,
        remote_file_name: str,
        destination: str,
        overwrite: bool = False,
        progress_bar: bool = True,
    ):
        # Note: WandB doesn't support progress bars for downloading
        del progress_bar
        import wandb
        import wandb.errors

        # using the wandb.Api() to support retrieving artifacts on ranks where
        # artifacts are not initialized
        api = wandb.Api()
        if not self.entity or not self.project:
            raise RuntimeError('get_file_artifact can only be called after running init()')

        # replace all unsupported characters with periods
        # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
        if ':' not in remote_file_name:
            remote_file_name += ':latest'

        new_remote_file_name = re.sub(r'[^a-zA-Z0-9-_\.:]', '.', remote_file_name)
        if new_remote_file_name != remote_file_name:
            warnings.warn((
                'WandB permits only alpha-numeric, periods, hyphens, and underscores in file names. '
                f"The file with name '{remote_file_name}' will be stored as '{new_remote_file_name}'."
            ))

        try:
            wandb_artifact = api.artifact('/'.join([self.entity, self.project, new_remote_file_name]))
        except wandb.errors.CommError as e:
            raise FileNotFoundError(f'WandB Artifact {new_remote_file_name} not found') from e
        with tempfile.TemporaryDirectory() as tmpdir:
            wandb_artifact_folder = os.path.join(tmpdir, 'wandb_artifact_folder/')
            wandb_artifact.download(root=wandb_artifact_folder)
            wandb_artifact_names = os.listdir(wandb_artifact_folder)
            # We only log one file per artifact
            if len(wandb_artifact_names) > 1:
                raise RuntimeError(
                    'Found more than one file in WandB artifact. We assume the checkpoint is the only file in the WandB artifact.',
                )
            wandb_artifact_name = wandb_artifact_names[0]
            wandb_artifact_path = os.path.join(wandb_artifact_folder, wandb_artifact_name)
            if overwrite:
                os.replace(wandb_artifact_path, destination)
            else:
                os.rename(wandb_artifact_path, destination)

    def post_close(self) -> None:
        import wandb

        # Cleaning up on post_close so all artifacts are uploaded
        if not self._enabled or wandb.run is None or self._is_in_atexit:
            # Don't call wandb.finish if there is no run, or
            # the script is in an atexit, since wandb also hooks into atexit
            # and it will error if wandb.finish is called from the Composer atexit hook
            # after it is called from the wandb atexit hook
            return

        exc_tpe, exc_info, tb = sys.exc_info()

        if (exc_tpe, exc_info, tb) == (None, None, None):
            wandb.finish(0)
        else:
            # record there was an error
            wandb.finish(1)


def _convert_to_wandb_image(image: Union[np.ndarray, torch.Tensor], channels_last: bool) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        if image.dtype == torch.float16 or image.dtype == torch.bfloat16:
            image = image.data.cpu().to(torch.float32).numpy()
        else:
            image = image.data.cpu().numpy()

    # Error out for empty arrays or weird arrays of dimension 0.
    if np.any(np.equal(image.shape, 0)):
        raise ValueError(f'Got an image (shape {image.shape}) with at least one dimension being 0! ')

    # Squeeze any singleton dimensions and then add them back in if image dimension
    # less than 3.
    image = image.squeeze()

    # Add in length-one dimensions to get back up to 3
    # putting channels last.
    if image.ndim == 1:
        image = np.expand_dims(image, (1, 2))
        channels_last = True
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
        channels_last = True

    if image.ndim != 3:
        raise ValueError(
            textwrap.dedent(
                f'''Input image must be 3 dimensions, but instead
                            got {image.ndim} dims at shape: {image.shape}
                            Your input image was interpreted as a batch of {image.ndim}
                            -dimensional images because you either specified a
                            {image.ndim + 1}D image or a list of {image.ndim}D images.
                            Please specify either a 4D image of a list of 3D images''',
            ),
        )
    assert isinstance(image, np.ndarray)
    if not channels_last:
        image = image.transpose(1, 2, 0)
    return image


def _convert_to_wandb_mask(mask: Union[np.ndarray, torch.Tensor], channels_last: bool) -> np.ndarray:
    mask = _convert_to_wandb_image(mask, channels_last)
    mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f'Mask must be a 2D array, but instead got array of shape: {mask.shape}')
    return mask


def _preprocess_mask_data(
    masks: dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]],
    channels_last: bool,
) -> dict[str, np.ndarray]:
    preprocesssed_masks = {}
    for mask_name, mask_data in masks.items():
        if not isinstance(mask_data, Sequence):
            mask_data = mask_data.squeeze()
            if mask_data.ndim == 2:
                mask_data = [mask_data]
        preprocesssed_masks[mask_name] = np.stack([_convert_to_wandb_mask(mask, channels_last) for mask in mask_data])
    return preprocesssed_masks


def _create_wandb_masks_generator(
    masks: dict[str, Union[np.ndarray, torch.Tensor, Sequence[Union[np.ndarray, torch.Tensor]]]],
    mask_class_labels: Optional[dict[int, str]],
    channels_last: bool,
):
    preprocessed_masks: dict[str, np.ndarray] = _preprocess_mask_data(masks, channels_last)
    for all_masks_for_single_example in zip(*list(preprocessed_masks.values())):
        mask_dict = {name: {'mask_data': mask} for name, mask in zip(masks.keys(), all_masks_for_single_example)}
        if mask_class_labels is not None:
            for k in mask_dict.keys():
                mask_dict[k].update({'class_labels': mask_class_labels})
        yield mask_dict
