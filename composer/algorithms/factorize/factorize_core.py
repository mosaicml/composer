# Copyright 2021 MosaicML. All Rights Reserved.

import dataclasses
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

FractionOrInt = Union[int, float]


@dataclasses.dataclass
class LowRankSolution:
    Wa: Optional[torch.Tensor] = None
    Wb: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    rank: int = -1
    nmse: float = 0


def clean_latent_dims(latent_dims: FractionOrInt, in_dims: int, out_dims: int) -> int:
    if latent_dims <= 1:  # fraction of input or output channels
        latent_channels = int(latent_dims * min(in_dims, out_dims))
        return max(1, latent_channels)
    return int(latent_dims)


def max_rank_with_possible_speedup(in_channels: int, out_channels: int, kernel_size: Optional[Tuple] = None) -> int:
    # TODO less naive cost model than counting multiply-adds
    fan_in = in_channels
    if kernel_size is not None:
        fan_in *= np.prod(kernel_size)
    return (fan_in * out_channels) / (fan_in + out_channels) - 1


def apply_solution_to_module_parameters(solution: LowRankSolution, module0: torch.nn.Module, module1: torch.nn.Module, transpose: bool) -> None:
    assert solution.bias is not None, "Can't apply unititalized solution!"
    assert solution.Wa is not None, "Can't apply unititalized solution!"
    assert solution.Wb is not None, "Can't apply unititalized solution!"

    with torch.no_grad():
        # first op always has no bias since adds no expressivity
        if module0.bias is not None:
            module0.bias = torch.nn.Parameter(torch.zeros(solution.rank, dtype=module0.bias.dtype))
        module1.bias.copy_(solution.bias)
        Wa = solution.Wa
        Wb = solution.Wb
        if transpose:
            Wa = torch.transpose(Wa, 0, 1)
            Wb = torch.transpose(Wb, 0, 1)
        module0.weight = torch.nn.Parameter(Wa)
        module1.weight = torch.nn.Parameter(Wb)


def _lstsq(A: torch.Tensor, B: torch.Tensor):
    if A.shape[0] != B.shape[0]:
        raise IndexError("A has different number of rows than B! " f"A.shape = {A.shape}, B.shape = {B.shape}")
    if len(A.shape) != 2:
        raise IndexError("A is not rank 2 tensor: has shape", A.shape)
    if len(B.shape) != 2:
        raise IndexError("B is not rank 2 tensor: has shape", A.shape)

    # TODO more intelligence regarding choice of lstsq `driver` arg
    return torch.linalg.lstsq(A, B).solution

def _nmse(Y: torch.Tensor, Y_hat: torch.Tensor):
    diffs = Y - Y_hat
    return (diffs * diffs).mean() / Y.var()


def _svd_initialize(Wa: torch.Tensor, Wb: torch.Tensor, k: int):
    if Wb is None:
        W = Wa
    else:
        W = Wa @ Wb

    # TODO rank k randomized svd if k small enough
    U, s, Vt = torch.linalg.svd(W, full_matrices=False)
    Wa = U[:, :k]
    Wb = Vt[:k]

    # scale matrices equally for numerical "load-balancing"
    s_sqrt = torch.sqrt(s[:k])  # s is already a vector, not mat
    Wa *= s_sqrt
    Wb *= s_sqrt.reshape(-1, 1)
    return Wa, Wb


def factorize(X: torch.Tensor,
              Y: torch.Tensor,
              Wa: torch.Tensor,
              Wb: Optional[torch.Tensor] = None,
              bias: Optional[torch.Tensor] = None,
              rank: Union[int, float] = .5,
              n_iters: int = 3) -> LowRankSolution:
    X = X.detach()
    Y = Y.detach()
    Wa = Wa.detach()
    Wb = Wb.detach() if Wb is not None else None
    if rank < 1:
        # fraction of input dimensionality (or current rank, if smaller)
        rank = min(int(rank * X.shape[1]), Wa.shape[1])
    k = rank

    ret = LowRankSolution()

    original_bias = None
    if bias is not None:
        original_bias = bias.detach()
        Y = Y - original_bias
        ret.bias = original_bias

    # if requested latent rank is greater than or equal to either
    # input rank or output rank, no point in factorizing; just
    # return a single matrix
    if k >= X.shape[1] or k >= Y.shape[1]:
        Wa = _lstsq(X, Y)
        ret.Wa = Wa
        ret.rank = -1
        return ret

    # if requested latent rank is greater than current latent rank,
    # just don't do the factorization
    if k >= Wa.shape[1]:
        ret.Wa = Wa
        ret.Wb = Wb
        ret.rank = -1
        return ret

    Wa, Wb = _svd_initialize(Wa, Wb, k)

    Ya = _lstsq(X, Y)
    for _ in range(n_iters):
        # update Wb
        Xb = X @ Wa
        Yb = Y
        Wb = _lstsq(Xb, Yb)

        # update Wa
        # We need to solve (AXB = Y) <=> (AX = B.I @ Y) not (AX = BY).
        # Since X and Y are constants, we can precompute pinv(A) @ Y.
        # We then have:
        #   pinv(A) @ A @ X @ B = pinv(A) @ Y
        #   (A.T@A).I @ A.T @ A @ X @ B = pinv(A) @ Y
        #   X @ B = pinv(A) @ Y
        #   Y.T @ pinv(A).T = B.T @ X.T
        # then we just solve for X.T:
        #   B.T @ X.T = Y.T @ pinv(A).T
        # also, note that pinv(A) @ Y = lstsq(A, Y); this makes sense;
        # means that targets for XB are the optimal coeffs mapping A to Y
        # also, observe that AXB = Y is using X and Y as variable to solve
        # for and targets, not the X and Y vars we have in this function
        Xa = Wb
        Wa_T = _lstsq(Xa.T, Ya.T)
        Wa = Wa_T.T

    ret.Wa = Wa
    ret.Wb = Wb
    ret.rank = k
    Y_hat = (X @ Wa) @ Wb

    bias = (Y - Y_hat).mean(dim=0)
    if original_bias is not None:
        bias += original_bias
    ret.bias = bias

    Y_hat += bias
    ret.nmse = _nmse(Y, Y_hat)

    return ret


def _activations_conv2d_to_mat(activations,
                               kernel_size,
                               padding=0,
                               padding_mode='zeros',
                               stride=1,
                               dilation=1,
                               groups=1):
    if np.max(stride) > 1:
        raise NotImplementedError(f"Stride != 1 not implemented; got {stride}")
    if np.max(dilation) > 1:
        raise NotImplementedError(f"Dilation != 1 not implemented; got {dilation}")
    if groups != 1:
        raise NotImplementedError(f"Groups != 1 not implemented; got {groups}")
    if np.max(padding) > 0 and padding_mode.lower() != 'zeros':
        activations = F.pad(activations, pad=padding, mode=padding_mode)
        padding = 0
    # always default to stride=1 to maximize amount of data we get here
    # TODO downsample in batch size dim or use stride > 1 if it looks like
    # materializing full matrix will OOM
    ret = F.unfold(activations, kernel_size=kernel_size, padding=padding)
    ret = ret.transpose(1, 2)  # batch_sz, n_positions, fan_in
    return ret.reshape(-1, ret.shape[2])  # batch_sz * n_positions, fan_in


def _weights_conv2d_to_mat(weights: torch.Tensor):
    return weights.reshape(weights.shape[0], -1).T  # fan_in, out_channels


def _mat_to_weights_conv2d(mat: torch.Tensor, kernel_size):
    if mat is None:
        return None
    w = mat.T  # fan_in, out_channels -> out_channels, fan_in
    # XXX(nchw) This might silently do the wrong thing with nhwc layout
    return w.reshape(w.shape[0], -1, *kernel_size)


def factorize_conv2d(inputs,
                     weightsA: torch.Tensor,
                     weightsB: Optional[torch.Tensor] = None,
                     rank: Union[int, float] = .5,
                     biasA: Optional[torch.Tensor] = None,
                     biasB: Optional[torch.Tensor] = None,
                     n_iters=3,
                     **conv2d_kwargs) -> LowRankSolution:
    inputs = inputs.detach()
    weightsA = weightsA.detach()

    kernel_size = weightsA.shape[2:]
    X_mat = _activations_conv2d_to_mat(inputs, kernel_size=kernel_size, **conv2d_kwargs)
    Wa = _weights_conv2d_to_mat(weightsA)
    # NOTE: we compute outputs ourselves, instead of having an arg for them,
    # since 1) we ignore input stride, and 2) any other intermediate ops
    # or other discrepancies between user's actual settings and args they pass
    # would either cause errors or silently mess up the regression
    Y_mat = (X_mat @ Wa)
    if biasA is not None:
        biasA = biasA.detach()
        Y_mat += biasA

    Wb = None
    if weightsB is not None:
        weightsB = weightsB.detach()
        Wb = _weights_conv2d_to_mat(weightsB)
        Y_mat = Y_mat @ Wb

        if biasB is not None:
            biasB = biasB.detach()
            Y_mat += biasB
    elif biasB is not None:
        # fail fast if user passes in inconsistent combination of args
        raise RuntimeError("Got biasB, but weightsB=None; cannot apply bias")

    ret = factorize(X_mat, Y_mat, Wa, Wb, rank=rank, n_iters=n_iters)

    # now we need to convert from two matrices to one kxk conv kernel and one
    # 1x1 conv kernel. Here's why the 2nd has to be a 1x1: if it were instead
    # k'xk' for some k' > 1, we would either be doing k'^2 as much work
    # for fixed embedding size at each pixel, or we'd be need to have the
    # intermediate embeddings be 1/k'^2 as large. In the latter case, we'd
    # lose a lot of representational capacity. Also, the first op has to match
    # the kernel size of the original conv or the shapes don't work out.
    ret.Wa = _mat_to_weights_conv2d(ret.Wa, kernel_size=kernel_size)
    ret.Wb = _mat_to_weights_conv2d(ret.Wb, kernel_size=(1, 1))

    return ret
