# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import xentropy_cuda  # from apex

try:
    import xentropy_cuda_inplace
except ImportError:
    xentropy_cuda_inplace = None


# https://github.com/NVIDIA/apex/blob/master/apex/contrib/xentropy/softmax_xentropy.py
class SoftmaxCrossEntropyLossFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, padding_idx=0, half_to_float=False, inplace_backward=False):
        losses, max_log_sum_exp = xentropy_cuda.forward(logits, labels, smoothing, half_to_float)
        losses.masked_fill_(labels == padding_idx, 0)
        ctx.save_for_backward(logits, max_log_sum_exp, labels)
        ctx.smoothing = smoothing
        ctx.padding_idx = padding_idx
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        logits, max_log_sum_exp, labels = ctx.saved_tensors
        if not grad_loss.is_contiguous():
            grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == ctx.padding_idx, 0)
        if not ctx.inplace_backward:
            grad_logits = xentropy_cuda.backward(grad_loss, logits, max_log_sum_exp, labels, ctx.smoothing)
        else:
            grad_logits = xentropy_cuda_inplace.inplace_backward(  # type: ignore
                grad_loss, logits, max_log_sum_exp, labels, ctx.smoothing)
        return grad_logits, None, None, None, None, None


class CrossEntropyLossApex(nn.Module):

    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0, inplace_backward=False):
        super().__init__()
        if reduction not in ['mean', 'none']:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        if inplace_backward and xentropy_cuda_inplace is None:
            raise ImportError('CUDA extension xentropy_cuda_inplace not found')
        self.inplace_backward = inplace_backward

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        is_half = input.dtype == torch.float16
        # SoftmaxCrossEntropyLoss implicitly casts to float
        loss = SoftmaxCrossEntropyLossFn.apply(input, target, self.label_smoothing, self.ignore_index, is_half,
                                               self.inplace_backward)
        return loss.mean() if self.reduction == 'mean' else loss
