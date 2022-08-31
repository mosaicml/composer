# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import math
from typing import Optional, Tuple, Union, cast

import numpy as np
import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from composer.algorithms.factorize.factorize_core import LowRankSolution, factorize_conv2d, factorize_matrix


def _clean_latent_size(latent_size: Union[int, float], in_size: int, out_size: int) -> int:
    if latent_size < 1:  # fraction of input or output channels
        latent_channels = int(latent_size * min(in_size, out_size))
        return max(1, latent_channels)
    return int(latent_size)


def _max_rank_with_possible_speedup(in_channels: int,
                                    out_channels: int,
                                    kernel_size: Optional[_size_2_t] = None) -> int:
    # TODO less naive cost model than counting multiply-adds
    fan_in = in_channels
    if kernel_size is not None:
        fan_in *= np.prod(kernel_size)
    breakeven = (fan_in * out_channels) / (fan_in + out_channels)
    return int(math.ceil(breakeven - 1))  # round down, or 1 lower if divides evenly


def factorizing_could_speedup(module: torch.nn.Module, latent_size: Union[int, float]):
    """Whether factorizing a module a given amount could possibly yield a benefit.

    This computation is based on the number of multiply-add operations involved
    in the module's current forward pass versus the number that would be involved
    if it were factorized into two modules using the specified latent size. The
    operations are assumed to be dense and of the same data type in all cases.

    Note that this function returning true does not guarantee a wall-clock
    speedup, since splitting one operation into two involves more data movement
    and more per-op overhead.

    Args:
        module (torch.nn.Module): A :class:`torch.nn.Conv2d`, :class:`torch.nn.Linear`,
            :class:`.FactorizedConv2d`, or :class:`.FactorizedLinear`.
        latent_size (int | float): number of channels (for convolution) or
            features (for linear) in the latent representation. Can be
            specified as either an integer > 1 or as float within ``[0, 1)``.
            In the latter case, the value is interpreted as a fraction of
            ``min(in_features, out_features)`` for a linear module or
            ``min(in_channels, out_channels)`` for a convolution.

    Returns:
        bool: A ``bool`` indicating whether the provided amount of factorization
            could accelerate the provided module. If ``module`` is not one of
            the allowed types, always returns ``False``, since there is no
            supported way to factorize that module.
    """
    if isinstance(module, _FactorizedModule):
        return module.should_factorize(latent_size)
    elif isinstance(module, torch.nn.Conv2d):
        if module.groups > 1:
            return False  # can't factorize grouped convolutions yet
        latent_size = _clean_latent_size(latent_size, module.in_channels, module.out_channels)
        max_rank = _max_rank_with_possible_speedup(module.in_channels,
                                                   module.out_channels,
                                                   kernel_size=cast(_size_2_t, module.kernel_size))
        return latent_size <= max_rank
    elif isinstance(module, torch.nn.Linear):
        latent_size = _clean_latent_size(latent_size, module.in_features, module.out_features)
        max_rank = _max_rank_with_possible_speedup(module.in_features, module.out_features)
        return latent_size <= max_rank
    else:
        return False


def _apply_solution_to_module_parameters(solution: LowRankSolution, module0: torch.nn.Module, module1: torch.nn.Module,
                                         transpose: bool) -> None:
    error_msg = "Can't apply unititalized solution!"
    assert solution.bias is not None, error_msg
    assert solution.Wa is not None, error_msg
    assert solution.Wb is not None, error_msg

    with torch.no_grad():
        # first op always has no bias since adds no expressivity
        if module0.bias is not None:
            assert isinstance(module0.bias, torch.Tensor)
            module0.bias = torch.nn.parameter.Parameter(
                torch.zeros(solution.rank, dtype=module0.bias.dtype).to(device=module0.bias.device))  # type: ignore
        assert isinstance(module1.bias, torch.Tensor)
        module1.bias.copy_(solution.bias)
        Wa = solution.Wa
        Wb = solution.Wb
        if transpose:
            Wa = torch.transpose(Wa, 0, 1)
            Wb = torch.transpose(Wb, 0, 1)
        module0.weight = torch.nn.parameter.Parameter(Wa.to(device=module0.weight.device))  # type: ignore
        module1.weight = torch.nn.parameter.Parameter(Wb.to(device=module1.weight.device))  # type: ignore


class _FactorizedModule(nn.Module, abc.ABC):

    def __init__(self, in_size: int, out_size: int, latent_size: Union[int, float], kernel_size: _size_2_t = 1):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.latent_size = _clean_latent_size(latent_size, in_size, out_size)
        self.kernel_size = kernel_size

    def _check_child_modules_present(self):
        assert hasattr(self, 'module0'), 'module0 must be set during child class __init__!'
        assert hasattr(self, 'module1'), 'module1 must be set during child class __init__!'
        assert isinstance(self.module0, torch.nn.Module)
        assert isinstance(self.module1, torch.nn.Module)

    def forward(self, input: torch.Tensor):  # type: ignore reportIncompatibleMethodOverride
        self._check_child_modules_present()
        ret = self.module0(input)  # type: ignore reportGeneralTypeIssues
        if self.module1 is not None:
            ret = self.module1(ret)  # type: ignore reportGeneralTypeIssues
        return ret

    def reset_parameters(self):
        self._check_child_modules_present()
        cast(torch.nn.Module, self.module0).reset_parameters()  # type: ignore reportGeneralTypeIssues
        cast(torch.nn.Module, self.module1).reset_parameters()  # type: ignore reportGeneralTypeIssues

    def set_rank(self, input: torch.Tensor, rank: int) -> None:
        """Makes the module factorize using a ``rank``-dimensional latent representation.

        ``rank`` can be large enough that the factorization increases the
        number of multiply-add operations, but not larger than the current
        latent rank.

        Args:
            input (torch.Tensor): Tensor that can be passed to the model's `forward()` method.
            rank (int): Dimensionality of the latent representation; this is the
                size of the vector space when factorizing linear modules and
                the number of channels for convolutional modules.

        Raises:
            ValueError:
                If ``rank`` is larger than the current latent rank.
        """
        if rank > self.latent_size:
            raise ValueError(f'Requested rank {rank} exceeds current rank {self.latent_size}')
        if rank == self.latent_size:
            return
        soln = self.solution_for_rank(input, rank)
        self.apply_solution(soln)

    def _clean_latent_size(self, latent_size: Union[int, float]):
        return _clean_latent_size(latent_size, self.in_size, self.out_size)

    def _max_rank_with_speedup(self):
        if hasattr(self, 'module1') and self.module1 is not None:
            # already factorized, so reducing rank at all helps
            return self.latent_size - 1
        else:
            # not factorized yet; has to factorize enough to be worthwhile
            return _max_rank_with_possible_speedup(self.in_size, self.out_size, kernel_size=self.kernel_size)

    def should_factorize(self, proposed_rank: Union[int, float]) -> bool:
        """Whether factorizing with a given rank would reduce the number of multiply-add operations."""
        proposed_rank = self._clean_latent_size(proposed_rank)
        return proposed_rank <= self._max_rank_with_speedup()

    @abc.abstractmethod
    def _create_child_modules(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """This is used to populate the self.module0 and self.module1 attributes; it's not part of __init__ because the
        logic to initialize them is subclass-specific and might depend on the shared logic in __init__"""
        ...

    @abc.abstractmethod
    def solution_for_rank(self, input: torch.Tensor, rank: int) -> LowRankSolution:
        """Returns a solution that :meth:`.apply_solution` can use to update the module's level of factorization.

        This is seperate from :meth:`set_rank` so that one can generate and assess
        many possible solutions for a given module before choosing one.

        Args:
            input (torch.Tensor): An input to the module used to optimize the solution's
                weights. The optimization seeks to preserve the module's
                input-output mapping as much as possible, subject to the
                specified rank constraint.
            rank (int): The number of dimensions in the latent space into which
                the input is mapped.

        Returns:
            solution:
                An object encapsulating the new parameters to be used and their
                associated mean squared error on the input
        """
        ...

    @abc.abstractmethod
    def apply_solution(self, solution: LowRankSolution) -> None:
        """Updates module's child modules to reflect the factorization solution.

        This *always* applies the solution and doesn't check whether
        using the solution is worthwhile.

        Args:
            solution (LowRankSolution): An object encapsulating the new
                parameters to be used and their associated mean squared error on
                the input for which they were optimized. Can be obtained using
                :meth:`.solution_for_rank`.
        """
        ...


class FactorizedConv2d(_FactorizedModule):
    """Factorized replacement for :class:`torch.nn.Conv2d`.

    Splits the conv2d operation into two smaller conv2d operations, which
    are executed sequentially with no nonlinearity in between. This first
    conv2d can be thought of as projecting the feature maps into a
    lower-dimensional space, similar to PCA. The second produces outputs
    of the same shape as the unfactorized version based on the embeddings
    within this lower-dimensional space. Note that "dimensionality" here
    refers to the number of channels, not the spatial extent or tensor rank.

    The first conv2d has a kernel size of ``kernel_size``, while the second
    one always has a kernel size of :math:`1 \\times 1`. For large kernel sizes, the
    lower-dimensional space can be nearly as large as
    ``min(in_channels, out_channels)`` and still yield a reduction in
    multiply-add operations. For kernels sizes of :math:`1 \\times 1`,
    the break-even point is a 2x reduction in channel count, similar to
    :class:`.FactorizedLinear`.

    See :func:`.factorize_conv2d` for more details.

    Args:
        in_channels (int): number of channels in the input image.
        out_channels (int): number of channels produced by the convolution.
        kernel_size (int | tuple): size of the convolving kernel.
        latent_channels (int | float, optional): number of channels in the
            latent representation produced by the first small convolution.
            Can be specified as either an integer > 1 or as float within
            ``[0, 1)``. In the latter case, the value is interpreted as a fraction
            of ``min(in_features, out_features)`` for each linear module and
            is converted to the equivalent integer value, with a minimum of 1.
            Default: ``.25``.
        **kwargs: other arguments to :class:`torch.nn.Conv2d` are supported
            and will be used with the first of the two smaller ``Conv2d``
            operations. However, ``groups > 1`` and ``dilation > 1`` are
            not currently supported.

    Raises:
        ValueError:
            If ``latent_channels`` is not small enough for factorization
            to reduce the number of multiply-add operations. In this regime,
            factorization is both slower and less expressive than a
            non-factorized operation. Setting
            ``latent_features`` to  :meth:`.max_allowed_latent_channels`
            or a smaller value is sufficient to avoid this.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 latent_channels: Union[int, float] = .25,
                 **kwargs):
        super().__init__(in_size=in_channels,
                         out_size=out_channels,
                         latent_size=latent_channels,
                         kernel_size=kernel_size)
        if kwargs.get('groups', 1) > 1:
            raise NotImplementedError('Factorizing grouped convolutions is not supported.')
        self.kwargs = kwargs
        # conv2d factorization code requires most Conv2d arguments, but
        # not boolean 'bias'
        self.convolution_kwargs = {k: v for k, v in kwargs.items() if k != 'bias'}
        self.module0, self.module1 = self._create_child_modules()

    def _create_child_modules(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if not self.should_factorize(self.latent_channels):
            raise ValueError(
                f'latent_channels {self.latent_size} is not small enough to merit factorization! Must be <= {self._max_rank_with_speedup()}'
            )

        # this one produces identical output as a regular Conv2d would,
        # except with fewer output channels
        conv0 = nn.Conv2d(self.in_channels,
                          self.latent_channels,
                          self.kernel_size,
                          bias=False,
                          **self.convolution_kwargs)
        # this one increases the number of output channels
        conv1 = nn.Conv2d(self.latent_channels, self.out_channels, kernel_size=1, bias=True)
        return conv0, conv1

    # wrap shared fields in read-only properties matching the torch conv module API
    @property
    def in_channels(self) -> int:
        """See :class:`torch.nn.Conv2d`."""
        return self.in_size

    @property
    def out_channels(self) -> int:
        """See :class:`torch.nn.Conv2d`."""
        return self.out_size

    @property
    def latent_channels(self) -> int:
        """The number of of output channels for the first convolution,
        which is also the number of input channels for the second convolution."""
        return self.latent_size

    def solution_for_rank(self, input: torch.Tensor, rank: int) -> LowRankSolution:
        weight0 = self.module0.weight
        bias0 = self.module0.bias
        weight1, bias1 = self.module1.weight, self.module1.bias

        assert (bias0 is None) or isinstance(bias0, torch.Tensor)
        assert isinstance(bias1, torch.Tensor)
        assert isinstance(weight0, torch.Tensor)
        assert isinstance(weight1, torch.Tensor)

        return factorize_conv2d(input, weight0, weight1, rank=rank, biasA=bias0, biasB=bias1, **self.convolution_kwargs)

    def apply_solution(self, solution: LowRankSolution):
        self.latent_size = solution.rank
        self.module0.out_channels = solution.rank
        self.module1.in_channels = solution.rank
        _apply_solution_to_module_parameters(solution, self.module0, self.module1, transpose=False)

    @staticmethod
    def max_allowed_latent_features(in_features: int, out_features: int, kernel_size: _size_2_t) -> int:
        """Returns the largest latent channel count that reduces the number of multiply-adds.

        Args:
            in_channels (int): number of channels in the input image
            out_channels (int): number of channels produced by the convolution
            kernel_size (int | tuple): size of the convolving kernel

        Returns:
            latent_channels: the largest allowable number of latent channels
        """
        return _max_rank_with_possible_speedup(in_features, out_features, kernel_size=kernel_size)

    @staticmethod
    def from_conv2d(module: torch.nn.Conv2d, module_ix: int = -1, **kwargs) -> FactorizedConv2d:
        conv = FactorizedConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=cast(_size_2_t, module.kernel_size),
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=((module.bias is not None) and (module.bias is not False)),
            **kwargs  # custom params
        )
        conv.reset_parameters()
        return conv


class FactorizedLinear(_FactorizedModule):
    """Factorized replacement for :class:`torch.nn.Linear`.

    Splits the linear operation into two smaller linear operations which
    are executed sequentially with no nonlinearity in between. This first
    linear operation can be thought of as projecting the inputs into a
    lower-dimensional space, similar to PCA. The second produces outputs
    of the same shape as the unfactorized version based on the embeddings
    within this lower-dimensional space.

    If the lower-dimensional space is less than half the size of the
    smaller of the input and output dimensionality, this factorization
    can reduce the number of multiply-adds necessary to compute the output.
    However, because larger matrix products tend to utilize the hardware
    better, it may take a reduction of more than 2x to get a speedup
    in practice.

    See :func:`.factorize_matrix` for more details.

    Args:
        in_features (int): Size of each input sample
        out_features (int): size of each output sample
        bias (bool, optional): If set to False, the layer will not learn an additive bias.
            Default: ``True``.
        latent_features (int | float, optional): Size of the latent space.
            Can be specified as either an integer > 1 or as a float within
            ``[0, 0.5)``. In the latter case, the value is interpreted as a fraction
            of ``min(in_features, out_features)``, and is converted to the
            equivalent integer value, with a minimum of 1. Default: ``.25``.

    Raises:
        ValueError:
            If ``latent_features`` is not small enough for factorization
            to reduce the number of multiply-add operations. In this regime,
            factorization is both slower and less expressive than a
            non-factorized operation. Setting
            ``latent_features < min(in_features, out_features) / 2`` or
            using :meth:`.max_allowed_latent_features` is sufficient to avoid
            this.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 latent_features: Union[int, float] = .25):
        super().__init__(in_size=in_features, out_size=out_features, latent_size=latent_features)
        self.bias = bias
        self.module0, self.module1 = self._create_child_modules()

    def _create_child_modules(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if not self.should_factorize(self.latent_size):
            raise ValueError(
                f'latent_features {self.latent_size} is not small enough to merit factorization! Must be <= {self._max_rank_with_speedup()}'
            )

        module0 = nn.Linear(in_features=self.in_features, out_features=self.latent_size, bias=False)
        module1 = nn.Linear(in_features=self.latent_size, out_features=self.out_features, bias=self.bias)
        return module0, module1

    # wrap shared fields in read-only properties matching the torch conv module API
    @property
    def in_features(self) -> int:
        """See :class:`torch.nn.Linear`."""
        return self.in_size

    @property
    def out_features(self) -> int:
        """See :class:`torch.nn.Linear`."""
        return self.out_size

    @property
    def latent_features(self) -> int:
        """The dimensionality of the space into which the input is
        projected by the first matrix in the factorization."""
        return self.latent_size

    def solution_for_rank(self, input: torch.Tensor, rank: int) -> LowRankSolution:
        assert isinstance(self.module0.weight, torch.Tensor)
        assert isinstance(self.module1.weight, torch.Tensor)
        assert isinstance(self.module1.bias, torch.Tensor)
        weight0 = torch.transpose(self.module0.weight, 0, 1)
        weight1 = torch.transpose(self.module1.weight, 0, 1)
        bias1 = self.module1.bias
        target = self(input)

        return factorize_matrix(input, target, weight0, weight1, bias=bias1, rank=rank)

    def apply_solution(self, solution: LowRankSolution) -> None:
        self.latent_size = solution.rank
        self.module0.out_features = solution.rank
        self.module1.in_features = solution.rank
        _apply_solution_to_module_parameters(solution, self.module0, self.module1, transpose=True)

    @staticmethod
    def max_allowed_latent_channels(in_features: int, out_features: int) -> int:
        """Returns the largest latent feature count that reduces the number of multiply-adds.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.

        Returns:
            int: The largest allowable number of latent features.
        """
        return _max_rank_with_possible_speedup(in_features, out_features)

    @staticmethod
    def from_linear(module: torch.nn.Linear, module_ix: int = -1, **kwargs) -> FactorizedLinear:
        ret = FactorizedLinear(in_features=module.in_features,
                               out_features=module.out_features,
                               bias=((module.bias is not None) and (module.bias is not False)),
                               **kwargs)
        ret.reset_parameters()
        return ret
