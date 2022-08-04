# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`~yahp.hparams.Hparams` used to construct the :class:`~composer.trainer.trainer.Trainer`."""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional

import yahp as hp

import composer
from composer.algorithms.algorithm_hparams_registry import algorithm_registry
from composer.core.algorithm import Algorithm
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger_hparams_registry import logger_registry
from composer.models.model_hparams import ModelHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams, model_registry
from composer.utils.object_store.object_store_hparams import ObjectStoreHparams, object_store_registry

__all__ = ['NLPTrainerHparams', 'GLUETrainerHparams']


@dataclasses.dataclass
class GLUETrainerHparams(hp.Hparams):
    """Finetuning Hparams class.

    Specifies arguments that should be applied as overrides to all GLUE finetuning tasks when using examples/glue/run_glue_trainer.py.

    Args:
        model (ComposerModel, optional): See :class:`.Trainer`.
        algorithms (List[Algorithm], optional): See :class:`.Trainer`.
        finetune_ckpts (List[str], optional): See :class:`.Trainer`.
        load_ignore_keys (List[str] | (Dict) -> None, optional): See :class:`.Trainer`.
        load_logger_destination (LoggerDestination, optional): See :class:`.TrainerHparams`.
        load_object_store (ObjectStoreHparams, optional): See :class:`.Trainer`.
        load_path (str, optional): See :class:`.Trainer`.
        loggers (List[LoggerDestination], optional): See :class:`.Trainer`.
        run_name (str, optional): See :class:`.Trainer`.
        save_folder (str, optional): See :class:`.Trainer`.

    Example:
        Specifying ``save_folder: path/to/example/folder`` in a yaml will force all glue tasks in composer/yamls/models/glue/ to
        save checkpoints to ``path/to/example/folder.``

    """
    model: Optional[ModelHparams] = hp.auto(Trainer, 'model')
    algorithms: Optional[List[Algorithm]] = hp.auto(Trainer, 'algorithms')
    finetune_ckpts: Optional[List[str]] = hp.optional(doc='list of checkpoints to finetune on', default=None)
    load_ignore_keys: Optional[List[str]] = hp.auto(Trainer, 'load_ignore_keys')
    load_logger_destination: Optional[LoggerDestination] = hp.auto(TrainerHparams, 'load_logger_destination')
    load_object_store: Optional[ObjectStoreHparams] = hp.auto(Trainer, 'load_object_store')
    load_path: Optional[str] = hp.auto(Trainer, 'load_path')
    loggers: Optional[List[LoggerDestination]] = hp.auto(Trainer, 'loggers')
    run_name: Optional[str] = hp.auto(Trainer, 'run_name')
    save_folder: Optional[str] = hp.auto(Trainer, 'save_folder')

    hparams_registry = {
        'algorithms': algorithm_registry,
        'load_object_store': object_store_registry,
        'load_logger_destination': logger_registry,
        'loggers': logger_registry,
        'model': model_registry,
    }


@dataclasses.dataclass
class NLPTrainerHparams(hp.Hparams):
    """Params for instantiating the :class:`.Trainer` for an NLP pretraining and finetuning job.

    .. seealso:: The documentation for the :class:`.Trainer`.

    Args:
        training_scheme (str): Training scheme to be used (one of "pretrain", "finetune", or "all"). Defaults to "all."
        pretrain_hparams (TrainerHparams, optional): Pretraining hyperparameters. Required if ``training_scheme`` is ``'pretrain'`` or ``'all'``.
        finetune_hparams (GLUETrainerHparams, optional): GLUE Finetuning shared hyperparameters.
    """

    # GLUE Specific Overrides test
    training_scheme: str = hp.optional(doc='training scheme used (one of "pretrain", "finetune", or "all")',
                                       default='all')
    pretrain_hparams: Optional[TrainerHparams] = hp.optional(doc='Pretraining hyperparameters', default=None)
    finetune_hparams: Optional[GLUETrainerHparams] = hp.optional(doc='GLUE Finetuning hyperparameters', default=None)

    @classmethod
    def load(cls, model: str) -> NLPTrainerHparams:
        """Load the NLPTrainerHparams for the given model."""
        model_hparams_file = os.path.join(
            os.path.dirname(composer.__file__),
            'yamls',
            'models',
            f'{model}.yaml',
        )
        trainer_hparams = NLPTrainerHparams.create(model_hparams_file, cli_args=False)
        return trainer_hparams
