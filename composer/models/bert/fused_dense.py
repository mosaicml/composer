# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/model/ops/fused_dense.py
# which was adapted from https://github.com/NVIDIA/apex/blob/master/apex/fused_dense/fused_dense.py
# On the backward pass, we don't use the fused kernel from cublasLt since that's a bit slower.
# Instead we use the regular backward from F.linear.
# We also make it work with pytorch amp.
# TD [2022-02-27] The fused backward is also less accurate, and it might silently fail to compute
# grad_bias (when it takes the cublas gemm code path instead of the cublasLt code path)
# TD [2022-04-19] The fused backward seems fine now. Also fused dense gelu dense works.

import fused_dense_lib as fused_dense_cuda
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd


# implements fused GEMM+bias in forward pass using mlp_cuda from apex
class FusedDenseFuncTD(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)  # type: ignore
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        output = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight, bias)
        return output.reshape(*batch_shape, output.shape[-1])

    @staticmethod
    @custom_bwd  # type: ignore
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_backward(
            x.reshape(batch_dim, n), weight, grad_output.reshape(batch_dim, grad_output.shape[-1]))
        return grad_input.reshape_as(x), grad_weight, grad_bias


fused_dense_function_td = FusedDenseFuncTD.apply


class FusedDenseTD(nn.Linear):

    def forward(self, x):
        if x.is_cuda and self.bias is not None:
            return fused_dense_function_td(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class FusedDenseResidualFunc(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)  # type: ignore
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        output = fused_dense_cuda.linear_bias_forward(x.reshape(batch_dim, n), weight, bias)
        return output.reshape(*batch_shape, output.shape[-1]), x

    @staticmethod
    @custom_bwd  # type: ignore
    def backward(ctx, grad_output, grad_input):
        x, weight = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = np.prod(batch_shape)
        grad_input, grad_weight, grad_bias = fused_dense_cuda.linear_bias_residual_backward(
            x.reshape(batch_dim, n), weight, grad_output.reshape(batch_dim, grad_output.shape[-1]),
            grad_input.reshape(batch_dim, n))
        return grad_input.reshape_as(x), grad_weight, grad_bias


fused_dense_residual_function = FusedDenseResidualFunc.apply


class FusedDenseResidual(nn.Linear):
    """Similar to FusedDense, but we return both the output and the input.
    This is so that in the backward pass, we can combine the input gradient from the residual branch
    with the input gradient from the matrix multiply, without having to do a separate addition.
    """

    def forward(self, x):
        if x.is_cuda and self.bias is not None:
            return fused_dense_residual_function(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias), x
