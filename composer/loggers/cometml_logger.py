# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Log to `Comet <https://www.comet.com/docs/v2/>`_."""

from __future__ import annotations

from typing import Any, Dict, Optional
from composer.core.state import State
from composer.loggers.logger import Logger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist
from composer.utils.import_helpers import MissingConditionalImportError

__all__ = ['CometMLLogger']


class CometMLLogger(LoggerDestination):
    """Log to `Comet <https://www.comet.com/docs/v2/>`_.

    Args:
        workspace (str, optional): The name of the workspace which contains the project
            you want to attach your experiment to. If nothing specified will default to your
            default workspace as configured at `www.comet.com <https://www.comet.com>`.
        project_name (str, optional): The name of the project to categorize your experiment in. 
            A new project with this name will be created under the Comet workspace if one
            with this name does not exist. If no project name specified, the experiment will go 
            under 'Uncategorized Experiments'. 
        log_code (bool): Whether to log your code in your experiment (default: ``False``).
        log_graph (bool): Whether to log your computational graph in your experiment
            (default: ``False``).
        name (str, optional): The name of your experiment. If not specified, it will be set
            to :attr:`.State.run_name`.
        rank_zero_only (bool, optional): Whether to log only on the rank-zero process.
            (default: ``False``).
        init_kwargs (Dict[str, Any], optional): Any additional init kwargs
            ``wandb.init`` (see
            `WandB do`cumentation <https://docs.wandb.ai/ref/python/init>`_).
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        project_name: Optional[str] = None,
        log_code: bool = False,
        log_graph: bool = False,
        name: Optional[str] = None,
        rank_zero_only: bool = True,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import comet_ml
        except ImportError as e:
            raise MissingConditionalImportError(extra_deps_group='comet_ml',
                                                conda_package='comet_ml',
                                                conda_channel='conda-forge') from e

        self._enabled = (not rank_zero_only) or dist.get_global_rank() == 0

        if init_kwargs is None:
            init_kwargs = {}

        if workspace is not None:
            init_kwargs['workspace'] = workspace

        if project_name is not None:
            init_kwargs['project_name'] = project_name

        init_kwargs['log_code'] = log_code
        init_kwargs['log_graph'] = log_graph


        self.name = name
        self._rank_zero_only = rank_zero_only
        self._init_kwargs = init_kwargs
        self.experiment = Optional[comet_ml.Experiment] = None

    def init(self, state: State, logger: Logger) -> None:
        import comet_ml
        del logger  # unused

        # Use the logger run name if the name is not set.
        if self.name is None:
            self.name = state.run_name

        # Adjust name and group based on `rank_zero_only`.
        if not self._rank_zero_only:
            self.name += f'-rank{dist.get_global_rank()}'

        if self._enabled:
            self.experiment = comet_ml.Experiment(**self._init_kwargs)
            self.experiment.set_name(self.name)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._enabled:
            self.experiment.log_metrics(dic=metrics, step=step)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        if self._enabled:
            self.experiment.log_parameters(hyperparameters)

