# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Comet <https://www.comet.com/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`_."""

from __future__ import annotations

from typing import Any, Dict, Optional

from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['CometMLLogger']


class CometMLLogger(LoggerDestination):
    """Log to `Comet <https://www.comet.com/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`_.

    Args:
        workspace (str, optional): The name of the workspace which contains the project
            you want to attach your experiment to. If nothing specified will default to your
            default workspace as configured in your comet account settings.
        project_name (str, optional): The name of the project to categorize your experiment in.
            A new project with this name will be created under the Comet workspace if one
            with this name does not exist. If no project name specified, the experiment will go
            under Uncategorized Experiments.
        log_code (bool): Whether to log your code in your experiment (default: ``False``).
        log_graph (bool): Whether to log your computational graph in your experiment
            (default: ``False``).
        name (str, optional): The name of your experiment. If not specified, it will be set
            to :attr:`.State.run_name`.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            (default: ``False``).
        exp_kwargs (Dict[str, Any], optional): Any additional kwargs to
            comet_ml.Experiment(see
            `Comet documentation <https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/?utm_source=mosaicml&utm_medium=partner&utm_campaign=mosaicml_comet_integration>`_).
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        log_code: bool = False,
        log_graph: bool = False,
        name: Optional[str] = None,
        rank_zero_only: bool = True,
        exp_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from comet_ml import Experiment
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='comet_ml',
                                                conda_package='comet_ml',
                                                conda_channel='conda-forge') from e

        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        if exp_kwargs is None:
            exp_kwargs = {}

        if workspace is not None:
            exp_kwargs['workspace'] = workspace

        if project_name is not None:
            exp_kwargs['project_name'] = project_name

        exp_kwargs['log_code'] = log_code
        exp_kwargs['log_graph'] = log_graph

        self.name = name
        self._rank_zero_only = rank_zero_only
        self._exp_kwargs = exp_kwargs
        self.experiment = None
        if self._enabled:
            self.experiment = Experiment(**self._exp_kwargs)
            self.experiment.log_other('Created from', 'mosaicml-composer')

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused

        # Use the logger run name if the name is not set.
        if self.name is None:
            self.name = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.name += f'-rank{dist.get_global_rank()}'

        if self._enabled:
            assert self.experiment is not None
            self.experiment.set_name(self.name)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            assert self.experiment is not None
            self.experiment.log_metrics(dic=metrics, step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            assert self.experiment is not None
            self.experiment.log_parameters(hyperparameters)

    def post_close(self):
        if self._enabled:
            assert self.experiment is not None
            self.experiment.end()
