# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`~yahp.hparams.Hparams` used to construct the :class:`~composer.trainer.trainer.Trainer`."""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional, Union

import yahp as hp

import composer
from composer.algorithms.algorithm_hparams_registry import algorithm_registry
from composer.core.algorithm import Algorithm
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger_hparams_registry import logger_registry
from composer.models.model_hparams import ModelHparams
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.object_store.object_store_hparams import ObjectStoreHparams, object_store_registry

__all__ = ['NLPTrainerHparams', 'GLUETrainerHparams']


@dataclasses.dataclass
class GLUETrainerHparams(hp.Hparams):
    """Finetuning Hparams class.

    Specifies arguments that should be applied as overrides to all GLUE finetuning tasks when using examples/run_glue_trainer.py.

    Args:
        model (ComposerModel, optional): The model to train. Can be user-defined or one of the models included
        with Composer.

            .. seealso:: :mod:`composer.models` for models built into Composer.
        algorithms (List[AlgorithmHparams], optional): The algorithms to use during training. (default: ``[]``)

            .. seealso:: :mod:`composer.algorithms` for the different algorithms built into Composer.
        finetune_ckpts (List[str], optional): List of load paths to checkpoints to run finetune jobs on.
        load_ignore_keys (List[str] | (Dict) -> None, optional): See :class:`.Trainer`.
        load_path (str, optional): See :class:`.Trainer`.
        load_object_store (ObjectStoreHparams, optional): See :class:`.Trainer`. Both ``load_logger_destination`` and
            ``load_object_store`` should not be provided since there can only be one location to load from.
        loggers (List[LoggerDestinationHparams], optional): Hparams for constructing the destinations
        to log to. (default: ``[]``)

            .. seealso:: :mod:`composer.loggers` for the different loggers built into Composer.
        run_name (str, optional): A name for this training run. If not specified, the timestamp will be combined with a
        :doc:`coolname <coolname:index>`, e.g. ``1654298855-electric-zebra``.
        save_folder (str, optional): See :class:`~composer.callbacks.checkpoint_saver.CheckpointSaver`.

    Example:
        Specifying ``save_folder: path/to/example/folder`` in a yaml will force all glue tasks in composer/yamls/models/glue/ to
        save checkpoints to ``path/to/example/folder.``

    """
    model: Optional[ModelHparams] = hp.auto(Trainer, 'model')
    algorithms: Optional[List[Algorithm]] = hp.auto(Trainer, 'algorithms')
    finetune_ckpts: Optional[List[str]] = hp.optional(doc='list of checkpoints to finetune on', default=None)
    load_ignore_keys: Optional[List[str]] = hp.auto(Trainer, 'load_ignore_keys')
    load_path: Optional[str] = hp.auto(Trainer, 'load_path')
    load_object_store: Optional[ObjectStoreHparams] = hp.auto(Trainer, 'load_object_store')
    loggers: Optional[List[LoggerDestination]] = hp.auto(Trainer, 'loggers')
    run_name: Optional[str] = hp.auto(Trainer, 'run_name')
    save_folder: Optional[str] = hp.auto(Trainer, 'save_folder')

    hparams_registry = {
        'algorithms': algorithm_registry,
        'load_object_store': object_store_registry,
        'loggers': logger_registry,
    }


@dataclasses.dataclass
class NLPTrainerHparams(hp.Hparams):
    """Params for instantiating the :class:`.Trainer` for an NLP pretraining and finetuning job.

    .. seealso:: The documentation for the :class:`.Trainer`.

    Args:
        pretrain_hparams (TrainerHparams): Pretraining hyperparameters
        finetune_hparams (GLUETrainerHparams, optional): GLUE Finetuning shared hyperparameters.
    """

    # GLUE Specific Overrides test
    pretrain_hparams: Optional[TrainerHparams] = hp.optional(doc='Pretraining hyperparameters', default=None)
    finetune_hparams: Optional[GLUETrainerHparams] = hp.optional(doc='GLUE Finetuning hyperparameters', default=None)
    training_scheme: Optional[str] = hp.optional(doc='training scheme used (one of "pretrain", "finetune", or "all")',
                                                 default='all')

    def initialize_object(self) -> Union[Trainer, None]:
        self.validate()

        if self.pretrain_hparams:
            return self.pretrain_hparams.initialize_object()

    @classmethod
    def load(cls, model: str) -> NLPTrainerHparams:
        model_hparams_file = os.path.join(
            os.path.dirname(composer.__file__),
            'yamls',
            'models',
            f'{model}.yaml',
        )
        trainer_hparams = NLPTrainerHparams.create(model_hparams_file, cli_args=False)
        assert isinstance(trainer_hparams, NLPTrainerHparams), 'trainer hparams should return an instance of self'
        return trainer_hparams
