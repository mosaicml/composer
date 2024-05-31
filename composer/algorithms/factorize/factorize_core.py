# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class LowRankSolution:
    """Bundles tensors used by a factorized linear operator.

    The factorization always splits the operator into two smaller linear
    operators. The first takes in input of the original shape and embeds it
    in a lower-dimensional space. The second maps this lower-dimensional space
    to the original output space.

    Args:
        Wa (torch.Tensor, optional): First linear operation in the
            factorized approximation. For a
            factorized linear operation, ``Wa`` is a matrix. For a factorized
            convolution, ``Wa`` matches the shape of the convolution's
            original weight parameter, except along the channel axis.
        Wb (torch.Tensor, optional): Second linear operation in the
            factorized approximation. Shape
            is such that composing ``Wb`` with ``Wb`` yields an output of
            the same size as the original operation.
        bias (torch.Tensor, optional): Vector added to the output of
            the second linear operation.
        rank (int, optional): Output dimensionality (channels or features) of
            the first linear operation, and input dimensionality of the second
            input operation. Default: ``-1``.
        nmse (float, optional): Normalized mean squared error obtained during
            the optimization procedure used to derive ``Wa``, ``Wb``, and
            ``bias``. This is equal to the raw mean squared error between
            the factorized approximation's output and the original output,
            divided by the variance of the original output. A value of 0
            means no error was introduced, and a value of 1 corresponds to
            capturing the output no better than chance. Default: ``0.0``.
    """
    Wa: Optional[torch.Tensor] = None
    Wb: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    rank: int = -1
    nmse: float = 0


def _lstsq(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.shape[0] != B.shape[0]:
        raise RuntimeError(f'A has different number of rows than B! A.shape = {A.shape}, B.shape = {B.shape}')
    if A.ndim != 2:
        raise RuntimeError('A is not a rank 2 tensor: has shape', A.shape)
    if B.ndim != 2:
        raise RuntimeError('B is not a rank 2 tensor: has shape', A.shape)

    # TODO more intelligence regarding choice of lstsq `driver` arg
    return torch.linalg.lstsq(A, B).solution


def _nmse(Y: torch.Tensor, Y_hat: torch.Tensor) -> float:
    diffs = Y.detach() - Y_hat.detach()
    return float((diffs * diffs).mean() / Y.var())


def _svd_initialize(Wa: torch.Tensor, Wb: Optional[torch.Tensor], k: int) -> tuple[torch.Tensor, torch.Tensor]:
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


def factorize_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    Wa: torch.Tensor,
    Wb: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    rank: Union[int, float] = .25,
    n_iters: int = 3,
) -> LowRankSolution:
    """Approximates a matrix by factorizing it into a product of two smaller matrices.

    Given a matrix ``W`` of shape ``[D, M]``, a bias vector of length ``M``,
    and a target rank ``rank < D``, returns a solution ``(Wa, Wb, new_bias)`` of
    tensors of shapes ``[N, rank]``, ``[rank, D]``, and ``M``, respectively.
    These tensors are chosen so as to minimize:

    :math:`||` ``Y - (X @ Wa @ Wb + new_bias)`` :math:`||_F`,

    where ``Y = X @ W + bias``, ``@`` denotes matrix multiplication,
    ``new_bias`` broadcasts along the row dimension,
    and :math:`||\\cdot||_F` denotes the sum of squared elements.
    In the case that rows of ``X`` correspond to samples
    from some distribution, this amounts to minimizing the expected mean
    squared error in the output.

    The input matrix can either be a single matrix ``W`` or a pair of matrices
    ``(Wa, Wb)``. The latter case corresponds to using a matrix ``W = Wa @ Wb``
    that has already been factorized and is supported in order to facilitate
    progressively decreasing the rank of the matrix.

    Args:
        X (torch.Tensor): Input used to evaluate the quality of the approximation.
            Shape is ``[N, D]``, where ``N`` is often the number of input samples and
            ``D`` is the dimensionality of each sample.
        Y (torch.Tensor): Output of applying the original matrix to ``X``.
            Must have shape ``[N, M]`` for some ``M``.
        Wa (torch.Tensor): Either the matrix to be factorized, or the first of the two smaller
            matrices in the already-factorized representation of this matrix.
            Must be of shape ``[D, M]`` in the former case and shape ``[D, d]``
            in the latter, for some ``d < D``.
        Wb (torch.Tensor, optional): If present, ``Wa`` is interpreted
            as the first of two smaller matrices, and ``Wb`` is taken to be the second.
            Must be of shape ``[d, M]``.
        bias (torch.Tensor, optional): A vector added to the output after
            performing the matrix product with X.
        rank (int | float, optional): the number of columns in the latent representation of X.
            Default: ``.25``.
        n_iters (int, optional): number of iterations used in the optimization process. Higher
            numbers yield lower mean squared error, though there are usually
            diminishing returns after a handful of iterations. Default: ``3``.

    Returns:
        LowRankSolution: A solution of rank ``rank`` that approximates the original convolution operation.
    """
    X = X.detach()
    Y = Y.detach()
    Wa = Wa.detach()
    Wb = Wb.detach() if Wb is not None else None
    if rank < 1:
        # fraction of input dimensionality (or current rank, if smaller)
        rank = min(int(rank * X.shape[1]), Wa.shape[1])
    k = int(rank)

    solution = LowRankSolution()

    original_bias = None
    if bias is not None:
        original_bias = bias.detach()
        Y = Y - original_bias
        solution.bias = original_bias

    # if requested latent rank is greater than or equal to either
    # input rank or output rank, factorization is counterproductive, so
    # return a single matrix
    if k >= X.shape[1] or k >= Y.shape[1]:
        Wa = _lstsq(X, Y)
        solution.Wa = Wa
        solution.rank = -1
        return solution

    # if requested latent rank is greater than current latent rank,
    # skip the factorization
    if k >= Wa.shape[1]:
        solution.Wa = Wa
        solution.Wb = Wb
        solution.rank = -1
        return solution

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

    solution.Wa = Wa
    solution.Wb = Wb
    solution.rank = k
    Y_hat = (X @ Wa) @ Wb

    bias = (Y - Y_hat).mean(dim=0)
    if original_bias is not None:
        bias += original_bias
    solution.bias = bias

    Y_hat += bias
    solution.nmse = _nmse(Y, Y_hat)

    return solution


def _activations_conv2d_to_mat(
    activations,
    kernel_size,
    padding=0,
    padding_mode='zeros',
    stride=1,
    dilation=1,
    groups=1,
):
    if np.max(stride) > 1:
        raise NotImplementedError(f'Stride != 1 not implemented; got {stride}')
    if np.max(dilation) > 1:
        raise NotImplementedError(f'Dilation != 1 not implemented; got {dilation}')
    if groups != 1:
        raise NotImplementedError(f'Groups != 1 not implemented; got {groups}')
    if np.max(padding) > 0 and padding_mode.lower() != 'zeros':
        if not isinstance(padding, list):
            padding = [padding]

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


def _mat_to_weights_conv2d(mat: Optional[torch.Tensor], kernel_size) -> Optional[torch.Tensor]:
    if mat is None:
        return None
    w = mat.T  # fan_in, out_channels -> out_channels, fan_in
    # XXX(nchw) This might silently do the wrong thing with nhwc layout
    return w.reshape(w.shape[0], -1, *kernel_size)


def factorize_conv2d(
    X: torch.Tensor,
    Wa: torch.Tensor,
    Wb: Optional[torch.Tensor] = None,
    rank: Union[int, float] = .25,
    biasA: Optional[torch.Tensor] = None,
    biasB: Optional[torch.Tensor] = None,
    n_iters=3,
    **conv2d_kwargs,
) -> LowRankSolution:
    """Approximates a :math:`K \\times K` convolution by factorizing it into a
    :math:`K \\times K` convolution  with fewer channels followed by a
    :math:`1 \\times 1` convolution.

    Given a convolutional weight tensor ``W`` for a 2d convolution of shape
    ``[out_channels, in_channels, k_h, k_w]`` and a vector ``bias`` of length
    ``out_channels``, returns a triple ``(Wa, Wb, new_bias)`` of
    tensors with shapes ``[rank, in_channels, k_h, k_w]``,
    ``[out_channels, rank, 1, 1]``, and ``[out_channels]``, respectively.
    ``Wa``, ``Wb``, and ``new_bias`` are chosen so as to minimize:

        :math:`||` ``(W * X + bias) - (Wb * (Wa * X) + new_bias)`` :math:`||_F`,

    where :math:`*` denotes convolution, ``bias`` broadcasts along all
    non-channel dimensions, and :math:`||\\cdot||_F` denotes the sum of
    squared elements.

    Similar to :func:`.factorize_matrix`, this function allows passing in an
    already-factorized weight tensor in order to enable progressive
    factorization. In this case, the single tensor ``W`` is replaced with
    a similar ``(Wa, Wb)`` pair as the output, though not necessarily with
    the same rank.

    Args:
        X (torch.Tensor): A tensor of shape ``[N, in_channels, H, W]``, for some
            ``N``, ``H``, and ``W``.
        Wa (torch.Tensor): The first weight tensor to convolve with ``X``. If
            ``Wb`` is not provided, must be of shape
            ``[out_channels, in_channels, k_h, k_w]``. Otherwise, must be of
            shape ``[original_rank, in_channels, k_h, k_w]`` for some
            ``original_rank < min(in_channels, out_channels)``.
        Wb (torch.Tensor, optional): The second weight tensor to convolve.
            with the input. If provided, must be of shape ``[out_channels, original_rank, 1, 1]``.
        rank (int | float, optional): number of channels in the latent representation of ``X``.
            Default: ``.25``.
        biasA (torch.Tensor, optional): Optional vector of biases. If ``Wb`` is
            ``None``, must have length ``out_channels``. Otherwise must have length
            ``original_rank``.
        biasB (torch.Tensor, optional): If provided, must have length ``out_channels``.
        n_iters (int, optional): number of iterations used in the optimization process.
            Higher numbers yield lower mean squared error, though there are usually
            diminishing returns after a handful of iterations. Default: ``3``.
        **conv2d_kwargs: Arguments such as ``padding``, ``stride``,
            ``dilation``, ``groups``, etc used in the original convolution. If
            these are not provided, the factorized tensors might not preserve
            the function computed by the original weight tensor as well.
            Note that not all combinations of arguments are supported.

    Returns:
        LowRankSolution: A solution of rank ``rank`` that approximates the original convolution operation.

    Raises:
        RuntimeError:
            If ``biasB`` is provided but not ``Wb`` is not.
        NotImplementedError:
            if ``conv2d_kwargs['dilation'] != 1`` or ``conv2d_kwargs['groups'] != 1``.
    """
    X = X.detach()
    Wa = Wa.detach()

    kernel_size = Wa.shape[2:]
    X_mat = _activations_conv2d_to_mat(X, kernel_size=kernel_size, **conv2d_kwargs)
    Wa = _weights_conv2d_to_mat(Wa)
    # NOTE: we compute outputs ourselves, instead of having an arg for them,
    # since 1) we ignore input stride, and 2) any other intermediate ops
    # or other discrepancies between user's actual settings and args they pass
    # would either cause errors or silently mess up the regression
    Y_mat = (X_mat @ Wa)
    if biasA is not None:
        biasA = biasA.detach()
        Y_mat += biasA

    if Wb is not None:
        Wb = Wb.detach()
        Wb = _weights_conv2d_to_mat(Wb)
        Y_mat = Y_mat @ Wb

        if biasB is not None:
            biasB = biasB.detach()
            Y_mat += biasB
    elif biasB is not None:
        # fail fast if user passes in inconsistent combination of args
        raise RuntimeError('Got biasB, but Wb=None; cannot apply bias')

    ret = factorize_matrix(X_mat, Y_mat, Wa, Wb, rank=rank, n_iters=n_iters)

    # now we need to convert from two matrices to one kxk conv kernel and one
    # 1x1 conv kernel. Here's why the 2nd has to be a 1x1: if it were instead
    # k'xk' for some k' > 1, we would either be doing k'^2 as much work
    # for fixed embedding size at each pixel, or we'd be need to have the
    # intermediate embeddings be 1/k'^2 as large. In the latter case, we'd
    # lose a lot of representational capacity. Also, the first op has to match
    # the kernel size of the original conv or the shapes don't work out.
    assert ret.Wa is not None
    ret.Wa = _mat_to_weights_conv2d(ret.Wa, kernel_size=kernel_size)
    ret.Wb = _mat_to_weights_conv2d(ret.Wb, kernel_size=(1, 1))

    return ret
