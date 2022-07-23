# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The :class:`~yahp.hparams.Hparams` used to construct the :class:`~composer.trainer.trainer.Trainer`."""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional, Tuple

import yahp as hp

import composer
from composer.algorithms.algorithm_hparams_registry import algorithm_registry
from composer.core.algorithm import Algorithm
from composer.loggers.logger_destination import LoggerDestination
from composer.loggers.logger_hparams_registry import logger_registry
from composer.trainer.trainer import Trainer
from composer.trainer.trainer_hparams import TrainerHparams
from composer.utils.object_store.object_store_hparams import ObjectStoreHparams, object_store_registry

__all__ = ['NLPTrainerHparams', 'GLUETrainerHparams']

@dataclasses.dataclass
class GLUETrainerHparams(hp.Hparams):
    """
    Finetuning Hparams class.

    Specifies arguments that should be broadcasted as overrides to all GLUE finetuning tasks when using examples/run_nlp_trainer.py.
    
    Args:
       ...
    """
    algorithms: Optional[List[Algorithm]] = hp.auto(Trainer, 'algorithms')
    load_ignore_keys: Optional[List[str]] = hp.auto(Trainer, 'load_ignore_keys')
    load_path: Optional[str] = hp.auto(Trainer, 'load_path')
    load_object_store: Optional[ObjectStoreHparams] = hp.auto(Trainer, 'load_object_store')
    loggers: Optional[List[LoggerDestination]] = hp.auto(Trainer, 'loggers')
    save_folder: Optional[str] = hp.auto(Trainer, 'save_folder')
    
    hparams_registry = {
        'algorithms': algorithm_registry,
        'load_object_store': object_store_registry,
        'loggers': logger_registry,
    }

    def initialize_object(self) -> Tuple: 
        load_object_store = None
        if self.load_object_store:
            load_object_store = self.load_object_store.initialize_object()

        return (self.algorithms, self.load_ignore_keys, self.load_path, load_object_store, self.loggers, self.save_folder)


@dataclasses.dataclass
class NLPTrainerHparams(hp.Hparams):
    """Params for instantiating the :class:`.Trainer` for an NLP pretraining and finetuning job.

    .. seealso:: The documentation for the :class:`.Trainer`.

    Args:
        pretrain_hparams (TrainerHparams): Pretraining hyperparameters
        finetune_hparams (GLUETrainerHparams, optional): GLUE Finetuning shared hyperparameters.
    """

    # GLUE Specific Overrides test
    pretrain_hparams: TrainerHparams = hp.optional(doc='Pretraining hyperparameters', default=None)
    finetune_hparams: Optional[GLUETrainerHparams] = hp.optional(doc='GLUE Finetuning hyperparameters', default=None)

    def validate(self):
        self.pretrain_hparams.validate()
       
    def initialize_object(self) -> Trainer:
        self.validate()

        # Glue Params
        if self.finetune_hparams:
            self.finetune_hparams = self.finetune_hparams.initialize_object()
        
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

load = NLPTrainerHparams.load
