# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Train models!"""

from __future__ import annotations

import contextlib
import datetime
import itertools
import logging
import os
import pathlib
import warnings
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple, Union, cast

import torch
import torch.distributed
import torch.utils.data
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import Metric, MetricCollection

from composer.callbacks import CheckpointSaver
from composer.core import (Algorithm, Callback, DataSpec, Engine, Evaluator, Event, Precision, State, Time, Timestamp,
                           ensure_data_spec, ensure_evaluator, ensure_time)
from composer.core.precision import get_precision_context
from composer.core.time import TimeUnit
from composer.core.types import Batch, BreakEpochException, PyTorchScheduler
from composer.loggers import Logger, LoggerDestination, LogLevel, ProgressBarLogger
from composer.models.base import ComposerModel
from composer.optim.decoupled_weight_decay import DecoupledSGDW
from composer.optim.scheduler import ComposerScheduler, compile_composer_scheduler
from composer.profiler import Profiler
from composer.trainer._deepspeed import _fix_batch_precision_for_deepspeed, _parse_deepspeed_config
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.trainer._scaler import ClosureGradScaler
from composer.trainer.ddp import DDPSyncStrategy, ddp_sync_context, prepare_ddp_module
from composer.trainer.devices import Device, DeviceCPU, DeviceGPU, DeviceTPU
from composer.utils import dist, ensure_tuple, format_name_with_dist, map_collection, module_surgery, reproducibility
from composer.utils.checkpoint import load_checkpoint, save_checkpoint
from composer.utils.file_helpers import GetFileNotFoundException
from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store import ObjectStore
import torch_xla.core.xla_model as xm

log = logging.getLogger(__name__)

__all__ = ["Trainer"]

# syntax to shorten the Scheduler type annoations
Scheduler = Union[ComposerScheduler, PyTorchScheduler]


def _scale_max_duration_by_ssr(
    scale_schedule_ratio: float,
    orig_max_duration: Optional[Time[int]],
) -> Optional[Time[int]]:
    if orig_max_duration is None:
        return None
    max_duration = cast(Time[int], orig_max_duration * scale_schedule_ratio)
    return max_duration


def _get_default_scheduler_frequency(schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]]):
    has_pytorch_scheduler = any(isinstance(scheduler, PyTorchScheduler) for scheduler in ensure_tuple(schedulers))
    if has_pytorch_scheduler:
        log.info()
        return TimeUnit.EPOCH
    else:
        return TimeUnit.BATCH


def _get_training_metrics(model: ComposerModel):
    train_metrics = model.metrics(train=True)
    if isinstance(train_metrics, Metric):

        train_metrics = MetricCollection([train_metrics])

    return train_metrics


def _compile_schedulers(
    schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]],
    state: State,
    scale_schedule_ratio: float,
) -> List[PyTorchScheduler]:
    compiled_schedulers = []
    for scheduler in ensure_tuple(schedulers):
        if isinstance(scheduler, PyTorchScheduler):
            scale_pytorch_scheduler(scheduler, scale_schedule_ratio)
            compiled_schedulers.append(scheduler)
        else:  # it's a composer scheduler
            compiled_schedulers.append(compile_composer_scheduler(scheduler, state, scale_schedule_ratio))

    return compiled_schedulers

def _is_adaptive_grad_accum(grad_accum: Union[int, str], device: Device):
    if grad_accum == 'auto':
        return True
    else:
        return False


def _get_initial_grad_accum(grad_accum: Union[int, str]):
    if grad_accum == "auto":
        return 1
    elif isinstance(grad_accum, int):
        return grad_accum
    else:
        raise ValueError("grad_accum must be an int or ``'auto'``")


def _is_cuda_oom(e: RuntimeError):
    return "CUDA out of memory" in str(e)


def _get_device(device: Optional[Union[str, Device]]):
    if not device:
        device = DeviceGPU() if torch.cuda.is_available() else DeviceCPU()
    elif isinstance(device, str):
        if device.lower() == 'cpu':
            device = DeviceCPU()
        elif device.lower() == 'gpu':
            device = DeviceGPU()
        elif device.lower() == 'tpu':
            device = DeviceTPU()
        else:
            raise ValueError(f'device ({device}) must be one of (cpu, gpu, tpu).')
    return device


def _distribute_and_get_random_seed(seed: Optional[int], device: Device):
    if not seed:
        seed = reproducibility.get_random_seed()

    if seed < 0 or seed > reproducibility.MAX_SEED:
        raise ValueError(f"Invalid seed: {seed}. It must be on [0; 2**32 - 1)")

    # using int64 to prevent overflow
    rank_zero_seed = device.tensor_to_device(torch.tensor([seed], dtype=torch.int64))
    dist.broadcast(rank_zero_seed, src=0)
    rank_zero_seed = rank_zero_seed.item()
    assert isinstance(rank_zero_seed, int)
    seed = rank_zero_seed + dist.get_global_rank()
    return rank_zero_seed, seed


def _get_ddp_sync_strategy(ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]], find_unused_parameters: bool):
    if ddp_sync_strategy is None:
        if find_unused_parameters:
            ddp_sync_strategy = DDPSyncStrategy.FORCED_SYNC
        else:
            ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC
    else:
        ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)
    return ddp_sync_strategy


class Trainer:
    def __init__(
        self,
        *,
        model: ComposerModel,
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = 'train',
        train_subset_num_batches: int = -1,
        compute_training_metrics: bool = False,
        max_duration: Optional[Union[int, str, Time]] = None,
        algorithms: Optional[Union[Algorithm, Sequence[Algorithm]]] = None,
        optimizers: Optional[torch.optim.Optimizer] = None,
        schedulers: Optional[Union[ComposerScheduler, PyTorchScheduler, Sequence[Union[ComposerScheduler,
                                                                                       PyTorchScheduler]]]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,
        eval_subset_num_batches: int = -1,

        callbacks: Optional[Union[Callback, Sequence[Callback]]] = None,
        loggers: Optional[Union[LoggerDestination, Sequence[LoggerDestination]]] = None,
        run_name: Optional[str] = None,
        progress_bar: bool = True,
        log_to_console: Optional[bool] = None,
        console_log_level: Union[LogLevel, str, Callable[[State, LogLevel], bool]] = LogLevel.EPOCH,
        console_stream: Union[str, TextIO] = 'stderr',

        load_path: Optional[str] = None,
        load_object_store: Optional[Union[ObjectStore, LoggerDestination]] = None,
        load_weights_only: bool = False,
        load_strict: bool = False,
        load_chunk_size: int = 1_048_576,
        load_progress_bar: bool = True,

        # Save Checkpoint
        save_folder: Optional[str] = None,
        save_filename: str = "ep{epoch}-ba{batch}-rank{rank}",
        save_artifact_name: str = "{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}",
        save_latest_filename: str = "latest-rank{rank}",
        save_latest_artifact_name: str = "{run_name}/checkpoints/latest-rank{rank}",
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = "1ep",
        save_weights_only: bool = False,
        save_num_checkpoints_to_keep: int = -1,

        # Graceful Resumption
        autoresume: bool = False,

        # DeepSpeed
        deepspeed_config: Optional[Dict[str, Any]] = None,

        # System/Numerics
        device: Optional[Union[str, Device]] = None,
        precision: Optional[Union[str, Precision]] = None,
        grad_accum: Union[int, str] = 1,

        # Reproducibility
        seed: Optional[int] = None,
        deterministic_mode: bool = False,

        # Distributed Training
        dist_timeout: float = 300.0,
        ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]] = None,

        # Grad Clip Norm
        grad_clip_norm: float = -1.0,

        # Profiling
        profiler: Optional[Profiler] = None,
    ):
        # Determine whether DeepSpeed is enabled
        deepspeed_enabled = deepspeed_config is not None

        # Device
        self._device = _get_device(device)

        # Distributed
        if deepspeed_enabled or dist.get_world_size() > 1:
            # distributed is always required with multi-rank training
            #if self._device is not "tpu":
            if False:
                dist.initialize_dist(self._device.dist_backend, datetime.timedelta(seconds=dist_timeout))


        #rank_zero_seed, seed = _distribute_and_get_random_seed(seed, self._device)

        reproducibility.seed_all(seed)
        if deterministic_mode:
            reproducibility.configure_deterministic_mode()

        # Precision
        if precision is None:
            precision = Precision.AMP if isinstance(device, DeviceGPU) else Precision.FP32
        if isinstance(precision, str):
            precision = Precision(precision)


        # optimizers and schedulers
        if not optimizers:
            optimizers = DecoupledSGDW(list(model.parameters()), lr=0.1)

        num_optimizers = len(ensure_tuple(optimizers))
        self.adaptive_gradient_accumulation = _is_adaptive_grad_accum(grad_accum, device=self._device)
        grad_accum = _get_initial_grad_accum(grad_accum)

        # Create the State
        self.state = State(
            rank_zero_seed=0,#rank_zero_seed,
            algorithms=algorithms,
            model=model,
            callbacks=callbacks,
            grad_accum=grad_accum,
            precision=precision,
            optimizers=optimizers,
        )

        # Profiler
        if profiler is not None:
            warnings.warn("The profiler is enabled. Using the profiler adds additional overhead when training.")
            self.state.profiler = profiler
            self.state.profiler.bind_to_state(self.state)

        # Console Logging
        loggers = list(ensure_tuple(loggers))
        if any(isinstance(x, ProgressBarLogger) for x in loggers):
            warnings.warn()
        else:
            loggers.append(
                ProgressBarLogger(
                    progress_bar=progress_bar,
                    log_to_console=log_to_console,
                    console_log_level=console_log_level,
                    stream=console_stream,
                ))

        # Logger
        self.logger = Logger(state=self.state, destinations=loggers, run_name=run_name)

        # Callbacks
        self.state.callbacks[:] = list(cast(List[Callback], loggers)) + self.state.callbacks

        # Checkpoint Saving
        self._checkpoint_saver = None
        if save_folder is not None:
            self._checkpoint_saver = CheckpointSaver(
                folder=save_folder,
                filename=save_filename,
                artifact_name=save_artifact_name,
                latest_filename=save_latest_filename,
                latest_artifact_name=save_latest_artifact_name,
                overwrite=save_overwrite,
                weights_only=save_weights_only,
                save_interval=save_interval,
                num_checkpoints_to_keep=save_num_checkpoints_to_keep,
            )
            self.state.callbacks.append(self._checkpoint_saver)

        # The Engine
        self.engine = Engine(state=self.state, logger=self.logger)

        # Set the logger
        model.logger = self.logger

        # Run Event.INIT
        self.engine.run_event(Event.INIT)

        import torch_xla.distributed.parallel_loader as pl
        
        self._train_data_spec = None if train_dataloader is None else ensure_data_spec(train_dataloader)
        if self._train_data_spec is not None:
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label,
                                      train_subset_num_batches)
            self.state.train_dataloader = self.state.dataloader
        self.train_metrics = _get_training_metrics(model) if compute_training_metrics else None

        # Max Duration
        if max_duration is not None:
            self.state.max_duration = ensure_time(max_duration, TimeUnit.EPOCH)

        #self.logger.data_fit({"rank_zero_seed": rank_zero_seed})

        assert isinstance(self.state.model, ComposerModel)
        self._original_model = self.state.model

        # Schedulers
        self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)
        if scale_schedule_ratio != 1.0:
            if len(self.state.schedulers) == 0:
                raise ValueError("Specifying `scale_schedule_ratio` without `schedulers` has no effect.")
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)

        if step_schedulers_every_batch is None:
            self._scheduler_step_frequency = _get_default_scheduler_frequency(schedulers)
        else:
            self._scheduler_step_frequency = TimeUnit.BATCH if step_schedulers_every_batch else TimeUnit.EPOCH


        # Grad Clip Norm
        self._grad_clip_norm = grad_clip_norm

        # Some algorithms require specific settings
        self._backwards_create_graph = any(map(lambda x: x.backwards_create_graph, ensure_tuple(algorithms)))
        self._find_unused_parameters = any(map(lambda x: x.find_unused_parameters, ensure_tuple(algorithms)))
        self._ddp_sync_strategy = _get_ddp_sync_strategy(ddp_sync_strategy, self._find_unused_parameters)

        # Load Checkpoint
        self._rng_state = None

        # Move the model and optimizers to the specified device
        if not self.deepspeed_enabled:
            host_model_params = self.state.model.parameters()
            self.state.model = self._device.module_to_device(self.state.model)
            device_model_params = self.state.model.parameters()

            # use surgery to update the parameters of the optimizers, now that the model is on the device
            # see https://pytorch.org/docs/stable/optim.html#constructing-it
            module_surgery.replace_params_in_optimizer(old_params=host_model_params,
                                                       new_params=device_model_params,
                                                       optimizers=self.state.optimizers)

            # Move any remaining optimizer parameters onto the device
            self.state.optimizers = map_collection(self.state.optimizers, self._device.optimizer_to_device)

            if dist.get_world_size() > 1:
                # Only wrap the module if required
                self.state.model = prepare_ddp_module(self.state.model, self._find_unused_parameters)

    @property
    def deepspeed_enabled(self):
        return self.state.is_model_deepspeed

    @property
    def saved_checkpoints(self) -> List[Tuple[Timestamp, List[pathlib.Path]]]:
        if self._checkpoint_saver is None:
            return []
        return self._checkpoint_saver.saved_checkpoints

    def fit(
        self,
        *,
        # Train Dataloader
        train_dataloader: Optional[Union[Iterable, DataSpec, Dict[str, Any]]] = None,
        train_dataloader_label: str = "train",
        train_subset_num_batches: Optional[int] = None,
        compute_training_metrics: Optional[bool] = None,

        # Timing
        duration: Optional[Union[int, str, Time[int]]] = None,
        reset_time: bool = False,

        # Schedulers
        schedulers: Optional[Union[ComposerScheduler, PyTorchScheduler, Sequence[Union[ComposerScheduler,
                                                                                       PyTorchScheduler]]]] = None,
        scale_schedule_ratio: float = 1.0,
        step_schedulers_every_batch: Optional[bool] = None,

        # Evaluation
        eval_dataloader: Optional[Union[Iterable, DataSpec, Evaluator, Sequence[Evaluator]]] = None,
        eval_subset_num_batches: int = -1,
        eval_interval: Union[int, str, Time, Callable[[State, Event], bool]] = 1,

        # Numerics
        grad_accum: Optional[Union[int, str]] = None,
        precision: Optional[Union[str, Precision]] = None,

        # Grad Clipping
        grad_clip_norm: Optional[float] = None,
    ):
        # Train Dataloader

        if xm.is_master_ordinal():
            xm.rendezvous('once')
        
        if train_dataloader is not None:
            self._train_data_spec = ensure_data_spec(train_dataloader)
            self.state.set_dataloader(self._train_data_spec.dataloader, train_dataloader_label)
            self.state.train_dataloader = self.state.dataloader

            ########### TPU
            import torch_xla.distributed.parallel_loader as pl
            #device = xm.xla_device()
            self.state.train_dataloader = pl.MpDeviceLoader(self.state.train_dataloader, self._device)
            
        if self._train_data_spec is None:
            _raise_missing_argument_exception("train_dataloader")
        if train_subset_num_batches is not None:
            self.state.dataloader_len = train_subset_num_batches
        if compute_training_metrics is not None:
            self.train_metrics = _get_training_metrics(self._original_model) if compute_training_metrics else None

        # Reset Time
        if reset_time:
            self.state.timestamp = Timestamp()

        # Max Duration
        if duration is not None:
            duration = ensure_time(duration, TimeUnit.EPOCH)
            self.state.max_duration = duration + self.state.timestamp.get(duration.unit)

        if self.state.max_duration is None:
            _raise_missing_argument_exception("max_duration")

        # Scale Schedule Ratio and Schedulers
        if scale_schedule_ratio != 1.0:
            self.state.max_duration = _scale_max_duration_by_ssr(scale_schedule_ratio, self.state.max_duration)
        if schedulers is not None:
            self.state.schedulers = _compile_schedulers(schedulers, self.state, scale_schedule_ratio)

            if step_schedulers_every_batch is None:
                self._scheduler_step_frequency = _get_default_scheduler_frequency(schedulers)
            else:
                self._scheduler_step_frequency = TimeUnit.BATCH if step_schedulers_every_batch else TimeUnit.EPOCH

        # Evaluators
        if eval_dataloader is not None:
            evaluators = [
                ensure_evaluator(evaluator, default_metrics=self._original_model.metrics(train=False))
                for evaluator in ensure_tuple(eval_dataloader)
            ]
            _set_evaluator_interval_and_subset_num_batches(
                evaluators=evaluators,
                eval_interval=eval_interval,
                subset_num_batches=eval_subset_num_batches,
            )
            self.state.evaluators = evaluators


        # Grad Accum
        if grad_accum is not None:
            self.adaptive_gradient_accumulation = _is_adaptive_grad_accum(grad_accum, device=self._device)
            self.state.grad_accum = _get_initial_grad_accum(grad_accum)

        # Grad Clip Norm
        if grad_clip_norm is not None:
            if self.deepspeed_enabled:
                raise ValueError("Changing the grad_clip_norm when using DeepSpeed is not supported.")
            self._grad_clip_norm = grad_clip_norm

        # Precision
        if precision is not None:
            if self.deepspeed_enabled:
                raise ValueError("Changing the precision when using DeepSpeed is not supported")
            precision = Precision(precision)
            #_validate_precision(precision, self._device, self.deepspeed_enabled)
            self.state.precision = precision


        self._train_loop()
            

    def close(self):
        """Shutdown the trainer.

        .. seealso:: :meth:`.Engine.close` for additional information.
        """
        self.engine.close()

    def _compute_and_log_metrics(self, dataloader_label: str, log_level: LogLevel, metrics: MetricCollection):
        computed_metrics = metrics.compute()
        self.logger.data(
            log_level=log_level,
            data={f'metrics/{dataloader_label}/{name}': val for (name, val) in computed_metrics.items()},
        )
        self.state.current_metrics[dataloader_label] = computed_metrics

    def _spin_dataloaders(self):
        for evaluator in self.state.evaluators:
            dataloader = evaluator.dataloader.dataloader
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(0)
            for _ in dataloader:
                break

        # spin the train dataloader's sampler to get to the state of the desired epoch
        dataloader = self.state.dataloader
        assert dataloader is not None, "train dataloader is set on state after FIT_START"
        for epoch in range(int(self.state.timestamp.epoch)):
            if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                dataloader.sampler.set_epoch(epoch)
            for _ in dataloader:
                break

    def _accumulate_samples_and_tokens_across_ranks(self, num_samples: int, num_tokens: int) -> Tuple[int, int]:
        tensor = self._device.tensor_to_device(torch.tensor([num_samples, num_tokens], dtype=torch.int))

        ### probably need something else here
        xm.all_reduce("sum", tensor, scale=1.0 / xm.xrt_world_size())

        #dist.all_reduce(tensor, reduce_operation="SUM")
        #return int(tensor[0].cpu().item()), int(tensor[1].cpu().item())
        return int(tensor[0].item()), int(tensor[1].item())

    def _train_loop(self) -> None:
        self.logger.data_fit({"trainer/algorithms": [str(algo) for algo in self.state.algorithms]})

        assert self.state.dataloader is not None, "dataloader is set in __init__() or fit()"
        assert self._train_data_spec is not None, "The train data spec is set in __init__() or fit()"

        self.engine.run_event(Event.FIT_START)

        self.state.scaler = ClosureGradScaler() if self._use_closures() else GradScaler()
        use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

        self._spin_dataloaders()

        if self.state.timestamp.batch_in_epoch == 0 and self._rng_state is not None:
            # only restore the rng state here if the step in the current epoch is zero.
            reproducibility.load_rng_state(self._rng_state)
            self._rng_state = None

        if self.train_metrics is not None:
            self.train_metrics = self._ensure_metrics_device_and_dtype(self.train_metrics)

        # Flag if the epoch finished early, so it can be tracked whether to run the epoch end events
        finished_epoch_early = False

        while self.state.timestamp < self.state.max_duration:
            try:
                self.state.model.train()

                if int(self.state.timestamp.batch_in_epoch) == 0:
                    self.engine.run_event(Event.EPOCH_START)
                    self.logger.data_epoch({"epoch": int(self.state.timestamp.epoch)})
                    if self.train_metrics is not None:
                        # reset the metrics before every epoch
                        self.train_metrics.reset()

                dataloader = self.state.dataloader
                if isinstance(dataloader, DataLoader) and isinstance(dataloader.sampler, DistributedSampler):
                    dataloader.sampler.set_epoch(int(self.state.timestamp.epoch))

                for batch_idx, self.state.batch in enumerate(self._iter_dataloader()):

                    # if resuming, skip dataloader forward to the minibatch index
                    if batch_idx < int(self.state.timestamp.batch_in_epoch):
                        # Restore the RNG state immediately before the next batch is yielded from the dataloader
                        if batch_idx + 1 == int(self.state.timestamp.batch_in_epoch) and self._rng_state is not None:
                            reproducibility.load_rng_state(self._rng_state)
                            self._rng_state = None
                        continue

                    self.state.batch = self._device.batch_to_device(self.state.batch)
                    self.state.batch = self._train_data_spec.device_transforms(self.state.batch)
                    self.state.batch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
                    self.state.batch_num_tokens = self._train_data_spec.get_num_tokens_in_batch(self.state.batch)

                    if self.train_metrics is not None:
                        self.state.model.eval()
                        with torch.no_grad():
                            for eval_microbatch in self._train_data_spec.split_batch(
                                    self.state.batch, self.state.grad_accum):
                                self.train_metrics.update(*self._original_model.validate(eval_microbatch))

                    self.state.model.train()

                    self.engine.run_event(Event.AFTER_DATALOADER)

                    ## might be different for tpu
                    total_num_samples, total_num_tokens = self.state.batch_num_samples, self.state.batch_num_tokens
                    '''
                    total_num_samples, total_num_tokens = self._accumulate_samples_and_tokens_across_ranks(
                        num_samples=self.state.batch_num_samples,
                        num_tokens=self.state.batch_num_tokens,
                    )
                    '''
                    

                    self.engine.run_event(Event.BATCH_START)
                    self.logger.data_batch({
                        "trainer/global_step": int(self.state.timestamp.batch),
                        "trainer/batch_idx": self.state.timestamp.batch_in_epoch.value,
                    })

                    total_loss = self._train_batch(use_grad_scaling)

                    if use_grad_scaling:
                        self.state.scaler.update()

                    if total_loss is not None:
                        if not isinstance(total_loss, torch.Tensor):
                            total_loss = self._device.tensor_to_device(torch.tensor([total_loss]))

                        # total_loss can be None if gradient scaling failed
                        dist.all_reduce(total_loss, reduce_operation="SUM")
                        full_loss = total_loss.cpu().item()
                        self.logger.data_batch({'loss/train': full_loss / dist.get_world_size()})

                    self.state.timestamp = self.state.timestamp.to_next_batch(
                        samples=total_num_samples,
                        tokens=total_num_tokens,
                    )

                    if self._scheduler_step_frequency == TimeUnit.BATCH:
                        for scheduler in self.state.schedulers:
                            scheduler.step()

                    if self.train_metrics is not None:
                        self._compute_and_log_metrics(
                            dataloader_label='train',
                            log_level=LogLevel.BATCH,
                            metrics=self.train_metrics,
                        )

                    self.engine.run_event(Event.BATCH_END)

                    for evaluator in self.state.evaluators:
                        if evaluator.eval_interval(self.state, Event.BATCH_END):
                            self.eval(
                                dataloader=evaluator.dataloader,
                                dataloader_label=evaluator.label,
                                subset_num_batches=evaluator.subset_num_batches,
                                metrics=evaluator.metrics,
                                log_level=LogLevel.BATCH,
                            )

                    self.engine.run_event(Event.BATCH_CHECKPOINT)

                    if self.state.timestamp >= self.state.max_duration:
                        finished_epoch_early = True
                        break

            except BreakEpochException:
                log.info(f'Skipping the rest of Epoch {int(self.state.timestamp.epoch)}')

            if not finished_epoch_early or self.state.dataloader_len == self.state.timestamp.batch_in_epoch:
                self.state.timestamp = self.state.timestamp.to_next_epoch()

                if self.train_metrics is not None:
                    self._compute_and_log_metrics(
                        dataloader_label='train',
                        log_level=LogLevel.EPOCH,
                        metrics=self.train_metrics,
                    )

                if self._scheduler_step_frequency == TimeUnit.EPOCH:
                    for scheduler in self.state.schedulers:
                        scheduler.step()

                self.engine.run_event(Event.EPOCH_END)

                for evaluator in self.state.evaluators:
                    if evaluator.eval_interval(self.state, Event.EPOCH_END):
                        self.eval(
                            dataloader=evaluator.dataloader,
                            dataloader_label=evaluator.label,
                            subset_num_batches=evaluator.subset_num_batches,
                            metrics=evaluator.metrics,
                            log_level=LogLevel.EPOCH,
                        )

                self.engine.run_event(Event.EPOCH_CHECKPOINT)


    def _train_batch(self, use_grad_scaling: bool):
        assert self._train_data_spec is not None, "The train data spec should be set on __init__ or fit()"

        # Retry until we successfully complete training and return loss
        while True:
            total_loss = None
            # Note: We use uint8 instead of bool as BOR is not supported on all torch.distributed backends
            should_handle_cuda_oom = 0
            caught_timeout_error = None
            try:
                assert self.state.scaler is not None
                microbatches = self._train_data_spec.split_batch(self.state.batch, self.state.grad_accum)
                import torch_xla.core.xla_model as xm
                
                if self.deepspeed_enabled:
                    total_loss = self._train_microbatches(microbatches)
                elif self._use_closures():
                    for optimizer in self.state.optimizers:
                        if use_grad_scaling:
                            total_loss = self.state.scaler.step(
                                optimizer, closure=lambda **kwargs: self._train_microbatches(microbatches, **kwargs))
                        else:
                            if True:#self._device == "tpu":
                                total_loss = xm.optimizer_step(optimizer) #, optimizer_args={"closure": lambda_closure, **kwargs})#: self._train_microbatches(microbatches, **kwargs).item()})
                            else:
                                total_loss = optimizer.step(
                                    closure=lambda **kwargs: self._train_microbatches(microbatches, **kwargs).item())
                              
                else:
                    total_loss = self._train_microbatches(microbatches)
                    for optimizer in self.state.optimizers:
                        if use_grad_scaling:
                            if True:#self._device == "tpu":
                                xm.optimizer_step(optimizer)
                            else:
                                self.state.scaler.step(optimizer)
                        else:
                            if True:#self._device == "tpu":
                                xm.optimizer_step(optimizer)
                            else:
                                optimizer.step()

                              
            except RuntimeError as e:
                if _is_cuda_oom(e):
                    should_handle_cuda_oom = 1
                elif "Timed out" in str(e):
                    caught_timeout_error = e
                else:
                    raise

            # Propagate across all ranks if any rank hit CUDA OOM
            should_handle_cuda_oom = self._device.tensor_to_device(
                torch.tensor([should_handle_cuda_oom], dtype=torch.uint8))

            return total_loss
            '''
            dist.all_reduce(should_handle_cuda_oom, reduce_operation="MAX")
            if int(should_handle_cuda_oom.item()) == 1:
                self._handle_cuda_oom()
            elif caught_timeout_error:
                raise caught_timeout_error
            else:
                # Otherwise, return calculated loss
                return total_loss
            '''

    def _train_microbatches(self, microbatches: Sequence[Batch], ddp_sync: bool = True):
        if ddp_sync or not isinstance(self.state.model, DistributedDataParallel):
            context = contextlib.nullcontext
        else:
            context = cast(Callable[[], ContextManager], self.state.model.no_sync)

        assert self._train_data_spec is not None

        with context():
            self.engine.run_event(Event.BEFORE_TRAIN_BATCH)

            assert self.state.optimizers is not None
            assert self.state.scaler is not None

            use_grad_scaling = self._use_grad_scaling(self.state.precision, self.state.scaler)

            if not self.deepspeed_enabled:
                for optimizer in self.state.optimizers:
                    optimizer.zero_grad()

            # tracker for gradient accumulation
            total_loss = self._device.tensor_to_device(torch.zeros(size=(1,)))
            current_batch_size = sum([self._train_data_spec.get_num_samples_in_batch(batch) for batch in microbatches])

            for microbatch_idx, self.state.batch in enumerate(microbatches):
                is_final_microbatch = microbatch_idx + 1 == len(microbatches)
                self._train_microbatch(use_grad_scaling, current_batch_size, total_loss, is_final_microbatch)

            # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
            if use_grad_scaling:
                for optimizer in ensure_tuple(self.state.optimizers):
                    self.state.scaler.unscale_(optimizer)

            # clip gradients if the magnitude is too large
            if not self.deepspeed_enabled and self._grad_clip_norm >= 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.state.model.parameters(),
                    max_norm=self._grad_clip_norm,
                )

            self.engine.run_event(Event.AFTER_TRAIN_BATCH)

            return total_loss

    def _train_microbatch(self, use_grad_scaling: bool, current_batch_size: int, total_loss: torch.Tensor,
                          is_final_microbatch: bool):
        assert self.state.scaler is not None
        assert self._train_data_spec is not None

        microbatch_num_samples = self._train_data_spec.get_num_samples_in_batch(self.state.batch)
        sync_context = contextlib.nullcontext() if self.deepspeed_enabled else ddp_sync_context(
            self.state,
            is_final_microbatch,
            self._ddp_sync_strategy,
        )

        with sync_context:
            # forward pass
            self.engine.run_event(Event.BEFORE_FORWARD)

            with get_precision_context(self.state.precision):
                self.state.outputs = self.state.model(self.state.batch)

            self.engine.run_event(Event.AFTER_FORWARD)

            # loss
            self.engine.run_event(Event.BEFORE_LOSS)

            with get_precision_context(self.state.precision):
                self.state.loss = self._original_model.loss(self.state.outputs, self.state.batch)

            if not self.deepspeed_enabled:
                for loss in ensure_tuple(self.state.loss):
                    loss.mul_(microbatch_num_samples / current_batch_size)
                    total_loss += loss.detach().clone()

            assert self.state.loss is not None
            self.engine.run_event(Event.AFTER_LOSS)

            # backward
            self.engine.run_event(Event.BEFORE_BACKWARD)

            if use_grad_scaling:
                self.state.loss = cast(torch.Tensor, self.state.scaler.scale(self.state.loss))

            if self.deepspeed_enabled:
                self.state.deepspeed_model.backward(self.state.loss)

                # This is the same loss scaling and reporting we skipped earlier.
                for loss in ensure_tuple(self.state.loss):
                    loss.mul_(microbatch_num_samples / current_batch_size)
                    total_loss += loss.detach().clone()
            else:
                for loss in ensure_tuple(self.state.loss):
                    loss.backward(create_graph=self._backwards_create_graph)

            self.engine.run_event(Event.AFTER_BACKWARD)

        if self.deepspeed_enabled:
            self.state.deepspeed_model.step()


    def _use_grad_scaling(self, precision: Union[str, Precision], scaler: Optional[GradScaler]) -> bool:
        if self.deepspeed_enabled:
            return False

        precision = Precision(precision)
        use_grad_scaling = precision == Precision.AMP

        return use_grad_scaling

    def _iter_dataloader(self):
        marker = None
        if self.state.profiler is not None:
            marker = self.state.profiler.marker(f"dataloader/{self.state.dataloader_label}", categories=["dataloader"])
        assert self.state.dataloader is not None, "the dataloader should be set before calling this method"

        if self.state.dataloader_len is None:
            dataloader_iter = iter(self.state.dataloader)
        else:
            dataloader_iter = itertools.islice(self.state.dataloader, int(self.state.dataloader_len))

        while True:
            if marker is not None:
                marker.start()
            try:
                yield next(dataloader_iter)
            except StopIteration:
                break
            finally:
                if marker is not None:
                    marker.finish()

    def _use_closures(self) -> bool:
        if self.deepspeed_enabled:
            return False

        if self.state.precision != Precision.AMP:
            return True

        return all(
            getattr(optimizer, "_step_supports_amp_closure", False)
            for optimizer in ensure_tuple(self.state.optimizers))

    def save_checkpoint(self, name: str = "ep{epoch}-ba{batch}-rank{rank}", *, weights_only: bool = False):
        return save_checkpoint(state=self.state, logger=self.logger, filename=name, weights_only=weights_only)
