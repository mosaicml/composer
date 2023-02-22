# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from composer import Algorithm, Event, State
from composer.core.precision import get_precision_context
from composer.core.time import Time
from composer.devices import Device, DeviceTPU
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.trainer.dist_strategy import prepare_fsdp_module
from composer.utils import get_device

log = logging.getLogger(__name__)

try:
    from transformers.modeling_outputs import \
        MaskedLMOutput  # TODO: This is not installed by default so it needs to made a conditional import
    mlm_installed = True
except ImportError:
    log.error('Unable to import transformers')
    mlm_installed = False

__all__ = ['Distillation', 'KLDivergence']


class Distillation(Algorithm):
    """`Knowledge Distillation <https://arxiv.org/abs/1503.02531>`_ trains a model with a
    modified loss function that includes the outputs from a supplementary "teacher" model.

    Args:
        teachers (dict, list, nn.Module) : A dictionary that contains path(s) to checkpoints and teacher
            models, of the form {``checkpoint_path``: ``model``}, where
            ``checkpoint_path`` is a string and ``model`` is a :class:`torch.nn.Module`.
        kd_loss_fn (Callable): Loss function with which to perform distillation.
            For example, :class:`torch.nn.MSELoss`, or :class:`.KLDivergence`.
        kd_loss_weight (float, optional): Weighting of kd loss in the overall loss.
            Default: ``0.9``.
        org_loss_weight (float, optional): Weighting of original loss. Default: ``0.1``.
        start_dur (float, optional): Fraction of training at which to start distillation.
            ``0`` is start and ``1.0`` is end. See :class:`composer.core.time.TimeUnit`. Default: ``0``.
        end_dur (float, optional): Fraction of training at which to end distillation.
            ``0`` is start and ``1.0`` is end. See :class:`composer.core.time.TimeUnit`.
            Default: ``1.0``.
        n_teachers_to_sample (int, optional): Number of teachers from which to (sub)
            sample for a single iteration. If ``n_teachers_to_sample`` is less than the
            number of teachers, teachers will be randomly subsampled on each iteration. We
            have found that randomly sampling one teacher from a group of teachers
            yields nearly as much benefit as ensembling the predictions of all the
            teachers. If ``None``, will default to be number of teachers (i.e.
            ``len(teachers)``).  Default: ``None``.
    """

    def __init__(
        self,
        teachers: Union[Dict[str, nn.Module], List[nn.Module], nn.Module],
        kd_loss_fn: Callable,
        kd_loss_weight: float = 0.9,
        org_loss_weight: float = 0.1,
        start_dur: float = 0.0,
        end_dur: float = 1.0,
        n_teachers_to_sample: Union[int, None] = None,
    ):
        super().__init__()
        self.teachers = []
        self.kd_loss_fn = kd_loss_fn
        self.base_kd_loss_weight = kd_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.org_loss_weight = org_loss_weight
        self.start_dur = start_dur
        self.end_dur = end_dur

        # teachers can be a dict of path / models, or a list of teachers with loaded
        # checkpoints, or a single teacher with loaded checkpoints
        if isinstance(teachers, Dict):
            for checkpoint_path, teacher in teachers.items():
                ckpt = torch.load(checkpoint_path)
                weights = None
                try:
                    weights = ckpt['state']['model']
                except KeyError:
                    log.error("Keys ['state']['model'] not found. Only Composer models supported at this time.")
                    raise
                try:
                    teacher.load_state_dict(weights)
                except:
                    log.error(f'Unable to load weights into teacher state_dict for checkpoint {checkpoint_path}.')
                    raise
                self.teachers.append(teacher)
        elif isinstance(teachers, nn.Module):
            self.teachers.append(teachers)
        else:
            self.teachers = teachers

        if n_teachers_to_sample is None:
            self.n_teachers_to_sample = len(self.teachers)
        else:
            self.n_teachers_to_sample = n_teachers_to_sample

    def match(self, event: Event, state: State) -> bool:
        if event == Event.FIT_START:
            return True
        elif event == Event.AFTER_LOSS:
            dur = state.get_elapsed_duration()
            assert isinstance(dur, Time)
            if dur >= self.start_dur and dur < self.end_dur:
                return True
            else:
                return False
        else:
            return False

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:

        if event == Event.FIT_START:
            # Handle parallelization
            # TODO: Figure out whether we even need DDP wrapping, given that teacher
            # params don't receive gradients
            for teacher in self.teachers:
                if state.fsdp_config is not None:
                    try:
                        prepare_fsdp_module(teacher,
                                            optimizers=None,
                                            fsdp_config=state.fsdp_config,
                                            precision=state.precision)
                    except:
                        log.error('Unable to prepare teacher model for FSDP')
                        raise
                else:
                    try:
                        self._move_teacher_model_to_device(teacher.module, state.model, state.device)
                    except:
                        log.error('Unable to move teacher.module to device. Will attempt to move teacher.')
                        try:
                            self._move_teacher_model_to_device(teacher, state.model, state.device)
                        except:
                            log.error('Unable to move teacher to device.')
                            raise

        elif event == Event.AFTER_LOSS:

            input = None
            if isinstance(state.batch, dict):
                state.batch.pop('labels')
                input = state.batch
            else:
                input = state.batch

            # Get teacher(s) to sample from
            t_idx = np.random.choice(range(len(self.teachers)), size=self.n_teachers_to_sample, replace=False)
            # get teacher output
            teacher_outputs = []

            # get teacher output
            with torch.no_grad():
                with get_precision_context(state.precision):
                    for idx in t_idx:
                        teacher_output = self.teachers[int(idx)](input)
                        if mlm_installed and isinstance(teacher_output, MaskedLMOutput):
                            teacher_outputs.append(teacher_output.logits.detach().clone())
                        else:
                            teacher_outputs.append(teacher_output)
            # Avg teacher outputs
            teacher_output = torch.mean(torch.stack(teacher_outputs), dim=0)

            # get original loss
            base_loss = state.loss

            # calculate KD loss
            student_output = state.outputs

            kd_loss = torch.empty(0)
            if isinstance(student_output, torch.Tensor):
                kd_loss = self.kd_loss_fn(teacher_output, student_output)
            elif mlm_installed:
                kd_loss = self.kd_loss_fn(teacher_output, student_output.logits)  #type: ignore #TODO: Fix this
            else:
                raise ValueError('student output is expected to be a Tensor')

            # Modify original loss and return to original format (dict, tuple, numeric)
            if isinstance(base_loss, tuple):
                state.loss = tuple([self.org_loss_weight * v for v in state.loss] + [kd_loss])
            elif isinstance(base_loss, dict):
                new_loss = {}
                for k, v in base_loss:
                    new_loss[k] = self.org_loss_weight * v
                new_loss['kd_loss'] = self.kd_loss_weight * kd_loss
                state.loss = new_loss  #type: ignore #TODO: Fix this
            elif isinstance(base_loss, torch.Tensor) or isinstance(base_loss, float):
                state.loss = self.org_loss_weight * base_loss + self.kd_loss_weight * kd_loss

    def _move_teacher_model_to_device(self, teacher_model: Union[ComposerModel, nn.Module, torch.Tensor],
                                      student_model: Union[ComposerModel, nn.Module], device: Device):
        """Moves the teacher model to student model device.

        Args:
            teacher_model (Union[ComposerModel, nn.Module, torch.Tensor]): Teacher model.
            student_model (ComposerModel): Student model.
            device (Device): Device to move teacher model to.
        """
        device = get_device(device)
        with torch.no_grad():
            # check if model is already on tpu
            if isinstance(device, DeviceTPU) and 'xla' not in str(next(student_model.parameters()).device):
                raise ValueError(
                    'Use model.to(xm.xla_device()) to set the model to the TPU before providing to the trainer.')
            else:
                assert isinstance(teacher_model, nn.Module)
                device.module_to_device(teacher_model)


class KLDivergence(nn.Module):
    """KL Divergence loss for distillation.

    Args:
        temperature (float, optional): Temperature for softmax. The temperature controls the entropy
            of the outputted probability distribution. As ``temperature`` approaches
            ``0``, the probability for the largest element approaches 1, and as
            ``temperature`` approaches ``inf``, the probability for each element will be
            the same. Default: ``4.0``.
        softmax_dim (int, optional): The dimension across which to compute the softmax.
            Tradition—and all theory and common sense—dictate that the only correct
            dimension across which to compute the softmax is the class dimension
            (``-1`` here). However, we found (by accident) that we obtained superior
            results when computing the softmax across tokens in a sequence (dimension =
            ``0``) for masked language modeling (MLM) pretraining. Accordingly, users may
            want to set ``softmax_dim=0`` when using distillation during MLM pretraining.
            Default: ``-1``.

    """

    def __init__(
        self,
        temperature: float = 4.0,
        softmax_dim: int = -1,
    ):
        super(KLDivergence, self).__init__()
        self.temperature = temperature
        self.softmax_dim = softmax_dim

    def forward(self, y_student: torch.Tensor, y_teacher: torch.Tensor):
        p_student = F.log_softmax(y_student / self.temperature, dim=self.softmax_dim)
        p_teacher = F.softmax(y_teacher / self.temperature, dim=self.softmax_dim)
        loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (self.temperature**2)
        return loss
