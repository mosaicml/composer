# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""FX-based model transformation and optimization.

Provides utilities to do FX-based model transformations.
"""

import logging
from typing import Any, Callable, List, Union

from torch.fx.graph_module import GraphModule

from composer.utils import ensure_tuple

log = logging.getLogger(__name__)

__all__ = ['count_op_instances', 'replace_op']


def count_op_instances(gm: GraphModule, ops: Union[Callable, str, List[Union[Callable, str]]]) -> int:
    """Counts the number of instances of ``op`` in ``gm``.

    .. rubric:: Example

    .. testsetup::

        import operator
        import torch
        from torch.fx import symbolic_trace
        from composer.utils.fx_utils import count_op_instances

    .. doctest::

        >>> class M(torch.nn.Module):
        ...   def forward(self, x, y):
        ...     return x + y, torch.add(x, y), x.add(y)
        >>> module = M()
        >>> traced = symbolic_trace(module)
        >>> count_op_instances(traced, torch.add)
        1
        >>> count_op_instances(traced, [operator.add, torch.add, "add"])
        3

    Arguments:
        module (GraphModule): The source FX-traced graph.
        op (Union[Callable, str, List[Union[Callable, str]]]):
            The operations to count.

    Returns:
        int: The number of instances of ``ops`` in ``gm``
    """
    ops = list(ensure_tuple(ops))
    return sum(any(n.target == op for op in ops) for n in gm.graph.nodes)


def replace_op(gm: GraphModule, src_ops: Union[Callable, str, List[Union[Callable, str]]],
               tgt_op: Callable[..., Any]) -> GraphModule:
    """Replace a single operator, torch method or function with another.

    .. rubric:: Example

    .. testsetup::

        import operator
        import torch
        from torch.fx import symbolic_trace
        from composer.utils.fx_utils import replace_op, count_op_instances

    .. doctest::

        >>> class M(torch.nn.Module):
        ...   def forward(self, x, y):
        ...     return x + y, torch.add(x, y), x.add(y)
        >>> module = M()
        >>> traced = symbolic_trace(module)
        >>> traced = replace_op(traced, [operator.add, torch.add, "add"], torch.mul)
        >>> count_op_instances(traced, torch.mul)
        3

    Arguments:
        module (GraphModule): The source FX-traced graph.
        src_ops (Union[Callable, str, List[Union[Callable, str]]):
            Replace these operations.
        tgt_op (Callable): Replacement for the operations

    Returns:
        GraphModule: Modified GraphModule with each instance of an op in ``src_ops`` replaced with
            ``tgt_op``. Returns the input if no instances are found.
    """
    src_ops = list(ensure_tuple(src_ops))
    for n in gm.graph.nodes:
        if any(n.target == op for op in src_ops):
            with gm.graph.inserting_after(n):
                new_node = gm.graph.call_function(tgt_op, n.args, n.kwargs)
                n.replace_all_uses_with(new_node)
            gm.graph.erase_node(n)
    gm.recompile()
    return gm


def detect_residual_pattern(gm: GraphModule):
    """Search and replace the pattern with another.

    Arguments:
        gm (GraphModule): The source FX-traced graph.

    Returns:
        GraphModule: Modified GraphModule.
    """
    raise NotImplementedError('detect_residual_pattern is currently not implemented.')


def replace_residual_with_stochastic(gm: GraphModule):
    """Replaces residual pattern with their stoachstic equivalent.

    Arguments:
        gm (GraphModule): The source FX-traced graph.

    Returns:
        GraphModule: Modified GraphModule.
    """
    raise NotImplementedError('replace_residual_with_stochastic is currently not implemented.')


def fuse_parallel_linears(gm: GraphModule):
    """If there are parallel linears in the model, fuse them together.

    Arguments:
        gm (GraphModule): The source FX-traced graph.

    Returns:
        GraphModule: Modified GraphModule.
    """
    raise NotImplementedError('fuse_parallel_linears is currently not implemented.')
