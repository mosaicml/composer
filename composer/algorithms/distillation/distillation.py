# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from composer import Algorithm, Event, State
from composer.core.precision import get_precision_context
from composer.core.time import Time
from composer.loggers import Logger
from composer.models import ComposerModel
from composer.utils import dist, get_file

log = logging.getLogger(__name__)


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
                self._move_teacher_model_to_device(self.teacher.module, state.model)
            except:
                log.error('Unable to move teacher.module to device. Will attempt to move teacher.')
                try:
                    self._move_teacher_model_to_device(self.teacher, state.model)
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
                    t_output = self.teacher(input)

            # get original loss
            base_loss = state.loss

            # calculate KD loss
            s_output = state.outputs

            kd_loss: torch.Tensor = None
            if isinstance(s_output, torch.Tensor):
                kd_loss = self.kd_loss_fn(t_output, s_output)
            else:
                kd_loss = self.kd_loss_fn(t_output.logits, s_output.logits)

            # modify original loss and return to original format (dict, tuple, numeric)
            if type(base_loss) is tuple:
                state.loss = tuple([self.org_loss * v for v in state.loss] + [kd_loss])
            elif type(base_loss) is dict:
                new_loss = dict()
                for k, v in base_loss:
                    new_loss[k] = self.org_loss_weight * v

                new_loss['kd_loss'] = self.kd_loss_weight * kd_loss

                state.loss = new_loss
            else:
                state.loss = self.org_loss_weight * base_loss + self.kd_loss_weight * kd_loss

    def _move_teacher_model_to_device(self, teacher_model: Union[ComposerModel, nn.Module, torch.Tensor],
                                      destination_model: Union[ComposerModel, nn.Module]):
        """Ensures the tensors of a teacher model are on the same device as a destination model."""
        with torch.no_grad():
            #device = next(destination_model.parameters()).device
            if torch.cuda.is_available():
                self.teacher.to(torch.cuda.current_device())
            # destination_params = destination_model.parameters()
            # teacher_params = teacher_model.parameters()
            # teacher_model.param_list = [s.to(d.device) for s, d in zip(teacher_params, destination_params)]

            # destination_buffers = destination_model.buffers()
            # teacher_buffers = teacher_model.buffers()
            # teacher_model.buffer_list = [s.to(d.device) for s, d in zip(teacher_buffers, destination_buffers)]

    #def _apply_teacher_transforms(self, )

    def _download_s3_weights(self):
        store = S3ObjectStore(self.s3_bucket_name)
        store.download_object(self.s3_path, '/tmp/teacher.pt', overwrite=True)


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
