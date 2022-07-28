# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Weights and Biases <https://wandb.ai/>`_."""

from __future__ import annotations

import atexit
import os
import pathlib
import re
import sys
import tempfile
import warnings
from typing import Any, Dict, List, Optional

from composer.core.state import State
from composer.loggers.logger import Logger, LogLevel
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['WandBLogger']


class WandBLogger(LoggerDestination):
    """Log to `Weights and Biases <https://wandb.ai/>`_.

    Args:
        project (str, optional): WandB project name.
        group (str, optional): WandB group name.
        name (str, optional): WandB run name.
            If not specified, the :attr:`.State.run_name` will be used.
        entity (str, optional): WandB entity name.
        tags (List[str], optional): WandB tags.
        log_artifacts (bool, optional): Whether to log
            `artifacts <https://docs.wandb.ai/ref/python/artifact>`_ (Default: ``False``).
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            When logging `artifacts <https://docs.wandb.ai/ref/python/artifact>`_, it is
            highly recommended to log on all ranks.  Artifacts from ranks ≥1 will not be
            stored, which may discard pertinent information. For example, when using
            Deepspeed ZeRO, it would be impossible to restore from checkpoints without
            artifacts from all ranks (default: ``False``).
        init_kwargs (Dict[str, Any], optional): Any additional init kwargs
            ``wandb.init`` (see
            `WandB documentation <https://docs.wandb.ai/ref/python/init>`_).
    """

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        log_artifacts: bool = False,
        rank_zero_only: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import wandb
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='wandb',
                                                conda_package='wandb',
                                                conda_channel='conda-forge') from e

        del wandb  # unused
        if log_artifacts and rank_zero_only and dist.get_world_size() > 1:
            warnings.warn(
                ('When logging artifacts, `rank_zero_only` should be set to False. '
                 'Artifacts from other ranks will not be collected, leading to a loss of information required to '
                 'restore from checkpoints.'))
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

    def _set_is_in_atexit(self):
        self._is_in_atexit = True

    def log_data(self, state: State, log_level: LogLevel, data: Dict[str, Any]):
        import wandb
        del log_level  # unused
        if self._enabled:
            wandb.log(data, step=int(state.timestamp.batch))

    def state_dict(self) -> Dict[str, Any]:
        import wandb

        # Storing these fields in the state dict to support run resuming in the future.
        if self._enabled:
            if wandb.run is None:
                raise ValueError('wandb must be initialized before serialization.')
            return {
                'name': wandb.run.name,
                'project': wandb.run.project,
                'entity': wandb.run.entity,
                'id': wandb.run.id,
                'group': wandb.run.group
            }
        else:
            return {}

    def init(self, state: State, logger: Logger) -> None:
        import wandb
        del logger  # unused

        # Use the logger run name if the name is not set.
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
            entity_and_project = [str(wandb.run.entity), str(wandb.run.project)]
            atexit.register(self._set_is_in_atexit)
        else:
            entity_and_project = [None, None]
        # Share the entity and project across all ranks, so they are available on ranks that did not initialize wandb
        dist.broadcast_object_list(entity_and_project)
        self.entity, self.project = entity_and_project
        assert self.entity is not None, 'entity should be defined'
        assert self.project is not None, 'project should be defined'

    def log_file_artifact(self, state: State, log_level: LogLevel, artifact_name: str, file_path: pathlib.Path, *,
                          overwrite: bool):
        del log_level, overwrite  # unused

        if self._enabled and self._log_artifacts:
            import wandb

            # Some WandB-specific alias extraction
            timestamp = state.timestamp
            aliases = ['latest', f'ep{int(timestamp.epoch)}-ba{int(timestamp.batch)}']

            # replace all unsupported characters with periods
            # Only alpha-numeric, periods, hyphens, and underscores are supported by wandb.
            new_artifact_name = re.sub(r'[^a-zA-Z0-9-_\.]', '.', artifact_name)
            if new_artifact_name != artifact_name:
                warnings.warn(('WandB permits only alpha-numeric, periods, hyphens, and underscores in artifact names. '
                               f"The artifact with name '{artifact_name}' will be stored as '{new_artifact_name}'."))

            extension = new_artifact_name.split('.')[-1]

            metadata = {f'timestamp/{k}': v for (k, v) in state.timestamp.state_dict().items()}
            # if evaluating, also log the evaluation timestamp
            if state.dataloader is not state.train_dataloader:
                # TODO If not actively training, then it is impossible to tell from the state whether
                # the trainer is evaluating or predicting. Assuming evaluation in this case.
                metadata.update({f'eval_timestamp/{k}': v for (k, v) in state.eval_timestamp.state_dict().items()})

            artifact = wandb.Artifact(
                name=new_artifact_name,
                type=extension,
                metadata=metadata,
            )
            artifact.add_file(os.path.abspath(file_path))
            wandb.log_artifact(artifact, aliases=aliases)

    def get_file_artifact(
        self,
        artifact_name: str,
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
        if ':' not in artifact_name:
            artifact_name += ':latest'

        new_artifact_name = re.sub(r'[^a-zA-Z0-9-_\.:]', '.', artifact_name)
        if new_artifact_name != artifact_name:
            warnings.warn(('WandB permits only alpha-numeric, periods, hyphens, and underscores in artifact names. '
                           f"The artifact with name '{artifact_name}' will be stored as '{new_artifact_name}'."))

        try:
            artifact = api.artifact('/'.join([self.entity, self.project, new_artifact_name]))
        except wandb.errors.CommError as e:
            if 'does not contain artifact' in str(e):
                raise FileNotFoundError(f'Artifact {new_artifact_name} not found') from e
            raise e
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_folder = os.path.join(tmpdir, 'artifact_folder')
            artifact.download(root=artifact_folder)
            artifact_names = os.listdir(artifact_folder)
            # We only log one file per artifact
            if len(artifact_names) > 1:
                raise RuntimeError(
                    'Found more than one file in artifact. We assume the checkpoint is the only file in the artifact.')
            artifact_name = artifact_names[0]
            artifact_path = os.path.join(artifact_folder, artifact_name)
            if overwrite:
                os.replace(artifact_path, destination)
            else:
                os.rename(artifact_path, destination)

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
