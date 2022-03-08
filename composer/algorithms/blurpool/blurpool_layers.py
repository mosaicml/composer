# Copyright 2021 MosaicML. All Rights Reserved.

# type: ignore
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


def _default_2d_filter():
    default_filter = np.array([[[
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]]]) * 1 / 16.0

    return torch.Tensor(default_filter)


def _padding_for_filt_2d_same(filt: torch.Tensor):
    _, _, h, w = filt.shape
    if h % 2 == 0:
        raise IndexError(f"Filter must have odd height; got {h}")
    if w % 2 == 0:
        raise IndexError(f"Filter must have odd width; got {w}")
    return (h // 2, w // 2)


def blur_2d(input: torch.Tensor, stride: _size_2_t = 1, filter: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a spatial low-pass filter.

    Args:
        input (torch.Tensor): a 4d tensor of shape NCHW
        stride (int or tuple, optional): stride(s) along H and W axes. If a single value is passed, this
            value is used for both dimensions.
        filter (torch.Tensor, optional): a 2d or 4d tensor to be cross-correlated with the input tensor
            at each spatial position, within each channel. If 4d, the structure
            is required to be ``(C, 1, kH, kW)`` where ``C`` is the number of
            channels in the input tensor and ``kH`` and ``kW`` are the spatial
            sizes of the filter.

    By default, the filter used is:

    .. code-block:: python

            [1 2 1]
            [2 4 2] * 1/16
            [1 2 1]

    Returns:
        The blurred input
    """
    _, c, h, w = input.shape
    n_in_channels = c

    if filter is None:
        filter = _default_2d_filter()
    if (filter.shape[0] == 1) and (n_in_channels > 1):
        # filt is already a rank 4 tensor
        filter = filter.repeat((n_in_channels, 1, 1, 1))

    _, _, filter_h, filter_w = filter.shape
    padding = _padding_for_filt_2d_same(filter)

    if h + 2 * padding[0] < filter_h:
        return input
    if w + 2 * padding[1] < filter_w:
        return input

    return F.conv2d(input, filter, stride=stride, padding=padding, groups=n_in_channels, bias=None)


def blurmax_pool2d(input: torch.Tensor,
                   kernel_size: _size_2_t = (2, 2),
                   stride: _size_2_t = 2,
                   padding: _size_2_t = 0,
                   dilation: _size_2_t = 1,
                   ceil_mode: bool = False,
                   filter: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Max-pooling with anti-aliasing.

    This is a nearly drop-in replacement for PyTorch's :func:`~torch.nn.functional.max_pool2d`.
    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.

    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.

    This function can be understood as decoupling the max from the pooling, and
    inserting a low-pass filtering step between the two. Concretely, this
    function computes the max within spatial neighborhoods of shape
    ``kernel_size``, then applies an anti-aliasing filter to smooth the maxes,
    and only then pools according to ``stride``.

    See also: :func:`~blur_2d`.

    Args:
        input (torch.Tensor): a 4d tensor of shape NCHW
        kernel_size (int or tuple, optional): size(s) of the spatial neighborhoods over which to pool.
            This is mostly commonly 2x2. If only a scalar ``s`` is provided, the
            neighborhood is of size ``(s, s)``.
        stride (int or tuple, optional): stride(s) along H and W axes. If a single value is passed, this
            value is used for both dimensions.
        padding (int or tuple, optional): implicit zero-padding to use. For the default 3x3 low-pass
            filter, ``padding=1`` (the default) returns output of the same size
            as the input.
        dilation (int or tuple, optional): amount by which to "stretch" the pooling region for a given
            total size. See :class:`~torch.nn.MaxPool2d`
            for our favorite explanation of how this works.
        ceil_mode (bool): when True, will use ceil instead of floor to compute the output shape
        filter (torch.Tensor, optional): a 2d or 4d tensor to be cross-correlated with the input tensor
            at each spatial position, within each channel. If 4d, the structure
            is required to be ``(C, 1, kH, kW)`` where ``C`` is the number of
            channels in the input tensor and ``kH`` and ``kW`` are the spatial
            sizes of the filter.

    By default, the filter used is:

    .. code-block:: python

            [1 2 1]
            [2 4 2] * 1/16
            [1 2 1]

    Returns:
         The blurred and max-pooled input
    """
    maxs = F.max_pool2d(input,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        ceil_mode=ceil_mode)
    return blur_2d(maxs, stride=stride, filter=filter)


class BlurMaxPool2d(nn.Module):
    """This module is a (nearly) drop-in replacement for PyTorch's :class:`~torch.nn.MaxPool2d`, but with an anti-
    aliasing filter applied.

    The only API difference is that the parameter ``return_indices`` is not
    available, because it is ill-defined when using anti-aliasing.

    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.

    See :func:`~blurmax_pool2d` for details.
    """

    # based on https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#MaxPool2d # noqa

    def __init__(self,
                 kernel_size: _size_2_t,
                 stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 ceil_mode: bool = False):
        super(BlurMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

        # we don't need this as part of state_dict, but making it a buffer
        # ensures that module.cuda(), module.to(), etc work out of the box
        self.register_buffer('filt2d', _default_2d_filter())

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
            ', dilation={dilation}, ceil_mode={ceil_mode}'.format(
                **self.__dict__)

    def forward(self, input: torch.Tensor):
        return blurmax_pool2d(input,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              dilation=self.dilation,
                              ceil_mode=self.ceil_mode,
                              filter=self.filt2d)

    @staticmethod
    def from_maxpool2d(module: torch.nn.MaxPool2d, module_index: int):
        return BlurMaxPool2d(kernel_size=module.kernel_size,
                             stride=module.stride,
                             padding=module.padding,
                             dilation=module.dilation,
                             ceil_mode=module.ceil_mode)


class BlurConv2d(nn.Module):
    """This module is a drop-in replacement for PyTorch's :class:`~torch.nn.Conv2d`, but with an anti-aliasing filter
    applied.

    The one new parameter is ``blur_first``. When set to ``True``, the
    anti-aliasing filter is applied before the underlying convolution, and
    vice-versa when set to ``False``. This mostly makes a difference when the
    stride is greater than one. In the former case, the only overhead is the
    cost of doing the anti-aliasing operation. In the latter case, the ``Conv2d``
    is applied with a stride of one to the input, and then the
    anti-aliasing is applied with the provided stride to the result. Setting
    the stride of the convolution to ``1`` can greatly increase the computational
    cost. E.g., replacing a stride of ``(2, 2)`` with a stride of ``1`` increases
    the number of operations by a factor of ``(2/1) * (2/1) = 4``. However,
    this approach most closely matches the behavior specified in the paper.

    This module should only be used to replace strided convolutions.

    See the associated `paper <http://proceedings.mlr.press/v97/zhang19a.html>`_
    for more details, experimental results, etc.

    See also: :func:`~blur_2d`.
    """

    # based partially on https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = None,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 blur_first: bool = True):

        super(BlurConv2d, self).__init__()
        self.blur_first = blur_first

        if self.blur_first:
            assert stride is not None
            conv_stride = stride
            self.blur_stride = 1
            blur_nchannels = in_channels
        else:
            conv_stride = 1
            self.blur_stride = kernel_size if (stride is None) else stride
            blur_nchannels = out_channels

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=conv_stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias)

        # this is the full 4d tensor we want; materialize it once, instead
        # of just-in-time during forward; we can do this in this class but
        # not the others because we know in_channels during __init__
        filt = _default_2d_filter().repeat(blur_nchannels, 1, 1, 1)
        self.register_buffer('blur_filter', filt)

    def forward(self, input: torch.Tensor):
        if self.blur_first:
            # blur in place, then apply (probably strided) conv
            # this is roughly the same number of flops as just applying
            # the original conv (though has some memory bandwidth cost)
            blurred = blur_2d(input, filter=self.blur_filter, stride=self.blur_stride)
            return self.conv.forward(blurred)
        else:
            # apply conv with stride of 1, then blur and (probably) downsample;
            # this is much more costly than a strided conv, at least in the
            # compute-bound regime
            activations = self.conv.forward(input)
            return blur_2d(activations, filter=self.blur_filter, stride=self.blur_stride)

    @staticmethod
    def from_conv2d(module: torch.nn.Conv2d, module_index: int = -1, blur_first: bool = True):
        has_bias = module.bias is not None and module.bias is not False
        blurconv = BlurConv2d(in_channels=module.in_channels,
                              out_channels=module.out_channels,
                              kernel_size=module.kernel_size,
                              stride=module.stride,
                              padding=module.padding,
                              dilation=module.dilation,
                              groups=module.groups,
                              bias=has_bias,
                              blur_first=blur_first)

        with torch.no_grad():
            blurconv.conv.weight.copy_(module.weight)
            if has_bias:
                blurconv.conv.bias.copy_(module.bias)

        return blurconv


class BlurPool2d(nn.Module):
    """This module just calls :func:`~blur_2d` in ``forward`` using the provided arguments."""

    def __init__(self, stride: _size_2_t = 2, padding: _size_2_t = 1) -> None:
        super(BlurPool2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.register_buffer('filt2d', _default_2d_filter())

    def forward(self, input: torch.Tensor):
        return blur_2d(input, stride=self.stride, filter=self.filt2d)
