# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import List, Optional, Sequence

from composer.benchmarker.benchmarker_hparams import BenchmarkerHparams
from composer.callbacks.timing_monitor import TimingMonitor
from composer.core.logging.base_backend import BaseLoggerBackend
from composer.datasets.hparams import DataloaderSpec
from composer.datasets.synthetic import SyntheticDataLabelType, SyntheticDataset
from composer.models.base import BaseMosaicModel
from composer.trainer.trainer import Trainer

NUM_PROFILING_STEPS = 500


class Benchmarker:

    def __init__(self,
                 model: BaseMosaicModel,
                 data_shape: Sequence[int],
                 total_batch_size: int,
                 grad_accum: int,
                 label_type: SyntheticDataLabelType = SyntheticDataLabelType.CLASSIFICATION_INT,
                 num_classes: Optional[int] = None,
                 label_shape: Optional[Sequence[int]] = None,
                 log_destinations: Optional[List[BaseLoggerBackend]] = None):

        dataset_size = total_batch_size * NUM_PROFILING_STEPS
        self.dataset = SyntheticDataset(total_dataset_size=dataset_size,
                                        data_shape=data_shape,
                                        num_unique_samples_to_create=total_batch_size,
                                        label_type=label_type,
                                        num_classes=num_classes,
                                        label_shape=label_shape)

        self.dataloader_spec = DataloaderSpec(dataset=self.dataset, shuffle=False, drop_last=True)

        # Default for now - adjust to work with algorithms
        timing_callback = TimingMonitor(min_steps=50, epoch_list=[0, 1], step_list=[0, 50], all_epochs=False)
        self.trainer = Trainer(
            model=model,
            train_dataloader_spec=self.dataloader_spec,
            eval_dataloader_spec=self.dataloader_spec,
            max_epochs=2,
            train_batch_size=total_batch_size,
            eval_batch_size=1,
            grad_accum=grad_accum,
            validate_every_n_epochs=100,  # don't validate
            log_destinations=log_destinations,
            callbacks=[timing_callback])

    def run_timing_benchmark(self):
        self.trainer.fit()

    @classmethod
    def create_from_hparams(cls, hparams: BenchmarkerHparams) -> Benchmarker:
        model = hparams.model.initialize_object()
        log_destinations = [l.initialize_object() for l in hparams.loggers]

        return cls(model=model,
                   data_shape=hparams.data_shape,
                   total_batch_size=hparams.total_batch_size,
                   grad_accum=hparams.grad_accum,
                   label_type=hparams.label_type,
                   num_classes=hparams.num_classes,
                   label_shape=hparams.label_shape,
                   log_destinations=log_destinations)