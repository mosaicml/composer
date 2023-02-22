# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor gradients during training."""

import collections

import torch

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist

__all__ = ['LossSpikeIntervention']

DEFAULT_TIMEOUT = 5_000
DEFAULT_LAYERWISE_LR_SCALE = 0.01
DEFAULT_CLEAR_OPT = True
DEFAULT_GLOBAL_LR_INCREASE = 1.0
DEFAULT_UNFREEZE_LAYERS = False

class MetricSpikeDetector:

    def __init__(self,
                 window_moving_average=25,
                 increase_factor=5,
                 increase_lookback=500,
                 plateau_min_duration=100,
                 end_spike_factor=1.10):

        self.window_moving_average = window_moving_average
        self.increase_factor = increase_factor
        self.plateau_min_duration = plateau_min_duration
        self.increase_lookback = increase_lookback
        self.fast_moving_average = collections.deque(maxlen=window_moving_average)
        self.intermediate_data_queue = collections.deque(maxlen=increase_lookback - window_moving_average)
        self.slow_moving_average = collections.deque(maxlen=increase_lookback)
        self.end_spike_factor = end_spike_factor
        self.in_spike = False
        self.mva_before_spike = None
        self.spike_batch_idx_start = None

    def insert_observation(self, obs, batch_idx):
        if len(self.fast_moving_average) >= self.fast_moving_average.maxlen:
            # move the oldest obs out of the fast moving average into the
            # intermediate data queue
            fast_obs = self.fast_moving_average.popleft()

            if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
                # move data from intermediate quque to slow MCVA queue
                intermediate_obs = self.intermediate_data_queue.popleft()
                self.slow_moving_average.append(intermediate_obs)

            self.intermediate_data_queue.append(fast_obs)

        self.fast_moving_average.append(obs)

        fast_mva = sum(self.fast_moving_average) / len(self.fast_moving_average)
        if not self.in_spike:
            if len(self.slow_moving_average) > self.window_moving_average:
                if self.mva_before_spike is None:
                    slow_mva = sum(self.slow_moving_average) / len(self.slow_moving_average)
                else:
                    slow_mva = self.mva_before_spike

                if fast_mva >= self.increase_factor * slow_mva:
                    self.in_spike = True
                    self.mva_before_spike = slow_mva
                    self.spike_batch_idx_start = batch_idx
        else:
            if batch_idx - self.spike_batch_idx_start > self.plateau_min_duration:
                # kill the layer!
                return True
            else:
                if fast_mva <= self.mva_before_spike * self.end_spike_factor:
                    self.in_spike = False
                    self.spike_batch_idx_start = None

        return False


class LossSpikeIntervention(Callback):

    def __init__(self,
                 metric='l2_norm/moment',
                 window_moving_average=25,
                 increase_factor=5,
                 increase_lookback=500,
                 plateau_min_duration=100,
                 end_spike_factor=1.10,
                 global_lr_scale=0.0,
                 unfreeze_policy=None):
        self.metric = metric
        self.global_lr_scale = global_lr_scale
        self.window_moving_average = window_moving_average
        self.increase_factor = increase_factor
        self.increase_lookback = increase_lookback
        self.plateau_min_duration = plateau_min_duration
        self.end_spike_factor = end_spike_factor

        self.metric_spike_detectors = {}
        self.frozen_layers = {}
        self.all_layers = set()
        self.unfreeze_policy = unfreeze_policy

    def fit_start(self, state: State, logger: Logger) -> None:
        for name, p in state.model.named_parameters():
            if p.requires_grad:
                self.all_layers.add(name)
                full_metric_name = f'{self.metric}/{name}'
                self.metric_spike_detectors[full_metric_name] = MetricSpikeDetector(
                    self.window_moving_average,
                    self.increase_factor,
                    self.increase_lookback,
                    self.plateau_min_duration,
                    self.end_spike_factor,
                )

    def unfreeze_layer(self, layer, layerwise_lr_scale, clear_opt, state):
        for name, p in state.model.named_parameters():
            if name == layer:
                p.requires_grad = True
                set_scaling = getattr(state.optimizers[0], 'set_scaling', None)
                if set_scaling:
                    set_scaling(p, layerwise_lr_scale)

                reset_param_state = getattr(state.optimizers[0], 'reset_param_state', None)
                if reset_param_state and clear_opt:
                    reset_param_state(p)
                break

    def unfreeze_layers(self, state, batch_idx):
        timeout = self.unfreeze_policy.get('timeout', DEFAULT_TIMEOUT)
        layerwise_lr_scale = self.unfreeze_policy.get('layerwise_lr_scale', DEFAULT_LAYERWISE_LR_SCALE)
        global_lr_increase = self.unfreeze_policy.get('global_lr_increase', DEFAULT_GLOBAL_LR_INCREASE)
        unfreeze_layers = self.unfreeze_layer.get('unfreeze_layers', DEFAULT_UNFREEZE_LAYERS)
        clear_opt = self.unfreeze_policy.get('clear_opt', DEFAULT_CLEAR_OPT)
        newly_unfrozen_layers = set()
        for layer in self.frozen_layers:
            if (batch_idx - self.frozen_layers[layer]) == timeout:
                # unfreeze the layer
                newly_unfrozen_layers.add(layer)

        if len(newly_unfrozen_layers) > 0:
            for optimizer in state.optimizers:
                for group in optimizer.param_groups:
                    group['lr'] *= global_lr_increase

            for scheduler in state.schedulers:
                scheduler.base_lrs = [global_lr_increase * lr for lr in scheduler.base_lrs]

        if unfreeze_layers:
            for layer in newly_unfrozen_layers:
                del self.frozen_layers[layer]
                self.all_layers.add(layer)
                full_metric_name = f'{self.metric}/{layer}'
                self.metric_spike_detectors[full_metric_name] = MetricSpikeDetector(
                        self.window_moving_average,
                        self.increase_factor,
                        self.increase_lookback,
                        self.plateau_min_duration,
                        self.end_spike_factor,
                )
                self.unfreeze_layer(layer, layerwise_lr_scale, clear_opt, state)

    def batch_end(self, state: State, logger: Logger):
        norm = 0.0
        optimizer_metrics = {}

        for name, p in state.model.named_parameters():
            if p.grad is not None and p.requires_grad:

                metric_reporter = getattr(state.optimizers[0], 'report_per_parameter_metrics', None)
                if callable(metric_reporter):
                    optimizer_metrics = metric_reporter(p, name, optimizer_metrics)

                if f'l2_norm/grad/{name}' not in optimizer_metrics:
                    param_grad_norm = torch.linalg.vector_norm(p.grad)
                    optimizer_metrics[f'l2_norm/grad/{name}'] = param_grad_norm

        if state.fsdp_enabled and dist.get_world_size() > 0:
            pre_reduce_metrics = getattr(state.optimizers[0], 'pre_reduce_metrics', None)
            if callable(pre_reduce_metrics):
                optimizer_metrics = pre_reduce_metrics(optimizer_metrics)

            dist_reduce_metrics = getattr(state.optimizers[0], 'dist_reduce_metrics', None)
            if callable(dist_reduce_metrics):
                optimizer_metrics = dist_reduce_metrics(optimizer_metrics)

        for metric in optimizer_metrics:
            if metric.startswith('l2_norm/grad'):
                norm += optimizer_metrics[metric]**2

        optimizer_metrics['l2_norm/grad/global'] = norm**0.5

        for metric in optimizer_metrics:
            if isinstance(optimizer_metrics[metric], torch.Tensor):
                optimizer_metrics[metric] = optimizer_metrics[metric].item()

        batch_idx = state.timestamp.batch.value
        newly_failed_layers = self.detect_failed_layers(optimizer_metrics, batch_idx)

        if len(newly_failed_layers) > 0:
            self.freeze_layers(newly_failed_layers, state)
            for optimizer in state.optimizers:
                for group in optimizer.param_groups:
                    group['lr'] *= self.global_lr_scale

            for scheduler in state.schedulers:
                scheduler.base_lrs = [self.global_lr_scale * lr for lr in scheduler.base_lrs]
            
            # we will decrease the rate at which we slow LR
            self.global_lr_scale *= 2

        if self.unfreeze_policy is not None:
            self.unfreeze_layers(state, batch_idx)

        optimizer_metrics['num_frozen_layers'] = len(self.frozen_layers)
        logger.log_metrics(optimizer_metrics)

        if len(self.all_layers) == 0:
            state.stop_training()

    def freeze_layers(self, newly_failed_layers, state):
        for layer in newly_failed_layers:
            self.all_layers.remove(layer)
            if layer not in self.frozen_layers:
                self.frozen_layers[layer] = state.timestamp.batch.value

        for name, p in state.model.named_parameters():
            if name in self.frozen_layers:
                p.requires_grad = False

    def detect_failed_layers(self, optimizer_metrics, batch_idx):
        newly_failed = []
        for logger_name, value in optimizer_metrics.items():
            if logger_name.startswith(self.metric):
                layer_name = logger_name.split('/')[-1]
                if layer_name in self.frozen_layers:
                    continue
                if self.metric_spike_detectors[logger_name].insert_observation(value, batch_idx):
                    newly_failed.append(layer_name)

        return newly_failed
