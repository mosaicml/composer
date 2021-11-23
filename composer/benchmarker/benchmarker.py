# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.distributed

from composer.benchmarker.benchmarker_hparams import BenchmarkerHparams
from composer.callbacks.benchmarker import Benchmarker as BenchmarkerCallback
from composer.core.logging.base_backend import BaseLoggerBackend
from composer.core.precision import Precision
from composer.datasets.hparams import DataloaderSpec
from composer.datasets.synthetic import SyntheticDataLabelType, SyntheticDataset
from composer.models.base import BaseMosaicModel
from composer.trainer.devices import DeviceCPU, DeviceGPU
from composer.trainer.trainer import Trainer

NUM_PROFILING_STEPS = 250  # needs to be high enough to be able to properly measure time


class Benchmarker:
    """Benchmarker for timing model training workloads with synthetic data.

    Args:
        model (BaseMosaicModel): The model to profile.
        data_shape (List[int]): Shape of the tensor for input samples.
        total_batch_size (int): The batch size to train with.
        grad_accum (int, optional): The number of microbatches to split a per-device batch into. Gradients
            are summed over the microbatches per device.
        label_type (SyntheticDataLabelType, optional), Type of synthetic data to create.
            If `CLASSIFICATION_INT` or `CLASSIFICATION_ONE_HOT` then `num_classes` must be specified.
            If `RANDOM_INT` then `label_shape` must be specified.
        num_classes (int, optional): Number of classes to use.
        label_shape (List[int]): Shape of the tensor for each sample label.
        log_destinations (List[BaseLoggerBackend], optional): The destinations to log training information to.
    """

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
        timing_callback = BenchmarkerCallback(min_steps=50, epoch_list=[0, 1], step_list=[0, 50], all_epochs=False)

        # Use optimal device settings based on what is available
        device = DeviceCPU()
        precision = Precision.FP32
        if torch.cuda.is_available() and (not torch.distributed.is_initialized() or
                                          torch.distributed.get_backend() == "nccl"):
            device = DeviceGPU(prefetch_in_cuda_stream=False)
            precision = Precision.AMP

        self.trainer = Trainer(
            model=model,
            train_dataloader_spec=self.dataloader_spec,
            eval_dataloader_spec=self.dataloader_spec,
            max_epochs=2,
            train_batch_size=total_batch_size,
            eval_batch_size=total_batch_size,
            grad_accum=grad_accum,
            device=device,
            precision=precision,
            validate_every_n_epochs=100,  # don't validate
            log_destinations=log_destinations,
            callbacks=[timing_callback])

    def run_timing_benchmark(self):
        """Run benchmarking to estimate throughput and wall clock time."""
        self.trainer.fit()

    @classmethod
    def create_from_hparams(cls, hparams: BenchmarkerHparams) -> Benchmarker:
        """Instantiate a Benchmarker using a `BenchmarkerHparams` object.

        Args:
            hparams (BenchmarkerHparams): The BenchmarkerHparams object used to instantiate the Benchmarker.

        Returns:
            A Benchmarker object initialized with the provided BenchmarkerHparams.
        """

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
