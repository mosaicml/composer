# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from composer import Algorithm, Event, State
from composer.core.precision import get_precision_context
from composer.core.time import Time
from composer.devices import Device, DeviceTPU
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.utils import get_device

log = logging.getLogger(__name__)

__all__ = ['Distillation', 'KLDivergence']


class Distillation(Algorithm):
    """`Knowledge Distillation <https://arxiv.org/abs/1503.02531>`_ trains a model with a
    modified loss function that includes the outputs from a supplementary "teacher" model.

    Args:
        teacher (nn.Module) : A teacher model.
        teacher_weights_path (str): Path to teacher weights.
        kd_loss_fn (Callable): Loss function with which to perform distillation.
            For example, :class:`torch.nn.MSELoss`, or :class:`.KLDivergence`.
        kd_loss_weight (float, optional): Weighting of kd loss in the overall loss.
            Default: ``0.9``.
        org_loss_weight (float, optional): Weighting of original loss. Default: ``0.1``.
        start_dur (float, optional): Fraction of training at which to start distillation.
            ``0`` is start and ``1.0`` is end. See :class:`composer.core.time.TimeUnit`. Default: ``0``.
        end_dur (float, optional): Fraction of training at which to end distillation.
            ``0`` is start and ``1.0`` is end. See :class:`composer.core.time.TimeUnit`. Default: ``1.0``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        teacher_weights_path: str,
        kd_loss_fn: Callable,
        kd_loss_weight: float = 0.9,
        org_loss_weight: float = 0.1,
        start_dur: float = 0.0,
        end_dur: float = 1.0,
    ):
        super().__init__()
        self.teacher = teacher
        self.teacher_weights_path = teacher_weights_path
        self.kd_loss_fn = kd_loss_fn
        self.base_kd_loss_weight = kd_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.org_loss_weight = org_loss_weight
        self.start_dur = start_dur
        self.end_dur = end_dur

        ckpt = torch.load(self.teacher_weights_path)
        weights = None
        try:
            weights = ckpt['state']['model']
        except KeyError:
            log.error("Keys ['state']['model'] not found. Only Composer models supported at this time.")
            raise
        try:
            self.teacher.load_state_dict(weights)
        except:
            log.error('Unable to load teacher checkpoint weights.')
            raise

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
            # move teacher to correct device after init
            try:
                self._move_teacher_model_to_device(self.teacher.module, state.model, state.device)
            except:
                log.error('Unable to move teacher.module to device. Will attempt to move teacher.')
                try:
                    self._move_teacher_model_to_device(self.teacher, state.model, state.device)
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

            # get teacher output
            with torch.no_grad():
                with get_precision_context(state.precision):
                    teacher_output = self.teacher(input)

            # get original loss
            base_loss = state.loss

            # calculate KD loss
            student_output = state.outputs

            kd_loss = torch.empty(0)
            if isinstance(student_output, torch.Tensor):
                kd_loss = self.kd_loss_fn(teacher_output, student_output)
            else:
                kd_loss = self.kd_loss_fn(teacher_output.logits, student_output.logits)  #type: ignore #TODO: Fix this

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
        p_teacher = F.softmax(y_teacher / self.T, dim=self.softmax_dim)
        loss = F.kl_div(p_student, p_teacher, size_average=False) * (self.temperature**2) / y_student.shape[0]
        return loss
