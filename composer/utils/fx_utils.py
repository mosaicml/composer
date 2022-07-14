# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""FX-based model transformation and optimization.

Provides utilities to do FX-based model transformations.
"""

import logging
import operator
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node
from torch.fx.passes.split_utils import split_by_tags

from composer.algorithms.stochastic_depth.stochastic_layers import BlockStochasticModule
from composer.utils import ensure_tuple

log = logging.getLogger(__name__)

__all__ = ['count_op_instances', 'replace_op', 'fuse_parallel_linears', 'apply_stochastic_residual']


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
    all_modules = dict(gm.named_modules())
    count = 0
    for n in gm.graph.nodes:
        for op in ops:
            if n.target == op:
                count += 1
            elif n.op == 'call_module' and isinstance(op, type) and isinstance(all_modules[n.target], op):
                count += 1
    return count


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


def _get_ancestors(node: Node) -> List[Node]:
    ancestorNodes = []
    while node.op != 'placeholder':
        ancestorNodes.append(node)
        node = node.all_input_nodes[0]
    return ancestorNodes


def _get_residual_block_nodes(nodeLHS: Node, nodeRHS: Node) -> Tuple[List[Node], List[Node]]:
    """Walk backwards from nodeLHS and nodeRSH to the root and construct lists of their parents.

    Arguments:
        nodeLHS (Node): left-hand side node for a binary operator
        nodeRHS (Node): right-hand side node for a binary operator

    Returns:
        (lhsAncestors, rhsAncestors): Two lists of nodes containing ancestors for ``nodeLHS`` and ``nodeRHS`` with
            their common ancestors removed.
    """
    lhsAncestors = _get_ancestors(nodeLHS)
    rhsAncestors = _get_ancestors(nodeRHS)

    # Iterate from back and eliminate common nodes
    while lhsAncestors and rhsAncestors and lhsAncestors[-1] == rhsAncestors[-1]:
        lhsAncestors.pop()
        rhsAncestors.pop()
    lhsAncestors.reverse()
    rhsAncestors.reverse()
    return lhsAncestors, rhsAncestors


def _attach_tag(nodes: List[Node], tag: str):
    """Attach tag to the given nodes for the splitter."""
    for node in nodes:
        node.tag = tag  # type: ignore[attr-defined]


def _tag_residual_nodes(gm: GraphModule) -> Tuple[List[str], int]:
    """Tag nodes for splitting."""
    # all nodes that are not a part of the residual blocks are tagged with "mainN_{count}".
    # a tag is required for all nodes by split_by_tags
    # Also an earlier tag can be repeated for later nodes.
    count = 0
    all_tags = []
    # In this pass over all nodes, we just tag them
    for node in gm.graph.nodes:
        default_tag = f'mainN_{count}'
        node.tag = default_tag
        if default_tag not in all_tags:
            all_tags.append(default_tag)
        if node.op == 'call_function' and node.target in [torch.add, operator.add]:
            assert len(node.all_input_nodes) == 2
            node0, node1 = node.all_input_nodes[0], node.all_input_nodes[1]
            lhs_nodes, rhs_nodes = _get_residual_block_nodes(node0, node1)
            if lhs_nodes or rhs_nodes:
                if len(lhs_nodes):
                    _attach_tag(lhs_nodes, f'non_res_{count}')
                    all_tags.append(f'non_res_{count}')
                if len(rhs_nodes):
                    _attach_tag(rhs_nodes, f'residual_{count}')
                    all_tags.append(f'residual_{count}')
                add_tag = f'addN_{count}'
                if add_tag not in all_tags:
                    all_tags.append(add_tag)
                node.tag = add_tag
                count += 1
    return all_tags, count


def _get_residual_modules(gm: GraphModule, node: Node) -> Tuple[Optional[GraphModule], Optional[GraphModule], int]:
    """Returns GraphModules for the main and residual branches.

    node.op is assumed to be a call_module
    """
    pattern = re.compile(r'non_res_(\d+)|residual_(\d+)')
    matches = pattern.match(str(node.target))
    if matches:
        idx = int(matches[1]) if matches[1] else int(matches[2])
        main_submod = getattr(gm, f'non_res_{idx}')
        residual_submod = getattr(gm, f'residual_{idx}', None)
        return main_submod, residual_submod, idx
    else:
        return None, None, 0


def _replace_residual_pattern(gm: GraphModule,
                              original_node: Node,
                              replacement_module: str,
                              has_residual_ops: bool = False) -> None:
    """Replaces main, residual and add_node with the ``replacement_module``.

    ``replacement_module`` is already added to the gm.
    """
    insert_node = original_node.prev
    add_node = original_node.next
    if has_residual_ops:
        add_node = original_node.next.next
    with gm.graph.inserting_after(insert_node):
        new_node = gm.graph.call_module(replacement_module, args=(insert_node,))  # type: ignore
        add_node.replace_all_uses_with(new_node)
        gm.graph.erase_node(add_node)
        if has_residual_ops:
            gm.graph.erase_node(original_node.next)
        gm.graph.erase_node(original_node)
    gm.graph.lint()


def apply_stochastic_residual(gm: GraphModule, drop_rate: float = 0.2) -> Tuple[GraphModule, int]:
    """Detect and replace residual pattern with their stochastic equivalent.

    Arguments:
        gm (GraphModule): The source FX-traced graph. It can be the whole model symbolically traced.

    Returns:
        GraphModule: Modified GraphModule that has stochastic residual connections.
    """
    if not isinstance(gm, GraphModule):
        raise ValueError(
            f'Input to apply_stochastic_residual should be an instance of GraphModule. Received {type(gm)}')
    all_tags, count = _tag_residual_nodes(gm)
    split_gm = split_by_tags(gm, all_tags)
    for node in split_gm.graph.nodes:
        if node.op != 'call_module':
            continue

        main_submod, residual_submod, idx = _get_residual_modules(split_gm, node)
        if main_submod:
            residual_st_instance = BlockStochasticModule(main_submod, residual_submod, drop_rate)
            split_gm.add_submodule(f'resi_st_{idx}', residual_st_instance)  # type: ignore
            _replace_residual_pattern(split_gm, node, f'resi_st_{idx}', residual_submod is not None)
    split_gm.recompile()
    return split_gm, count


def _can_linears_be_fused(linear_nodes: List[Node], all_modules: Mapping[str, nn.Module]) -> bool:
    """Check if all the linears have bias."""
    # Forcing node.target to str is fine here as we are dealing with nn.Modules
    # and their target is a str.
    bias = all_modules[str(linear_nodes[0].target)].bias is None

    return all(bias == (all_modules[str(node.target)].bias is None) for node in linear_nodes)


def _create_fused_linear(linear_nodes: List[Node],
                         all_modules: Mapping[str, nn.Module],
                         keep_weights: bool = False) -> Tuple[nn.Module, List[int]]:
    """Check if the linears can be fused.

    If the linears can be fused, create a fused nn.Linear instance and return it.
    """
    if keep_weights:
        raise NotImplementedError('This feature is currently not implemented.')

    assert len(linear_nodes) > 1, 'There should be at least 2 linears for fusion'
    out_features = []
    in_features = all_modules[str(linear_nodes[0].target)].in_features
    bias = all_modules[str(linear_nodes[0].target)].bias is not None

    for node in linear_nodes:
        out_features.append(all_modules[str(node.target)].out_features)
        assert in_features == all_modules[str(node.target)].in_features, 'mismatch in number of input features'
        assert bias == (all_modules[str(node.target)].bias is not None), 'mismatch in bias'

    return nn.Linear(in_features, sum(out_features), bias=bias), out_features  # type: ignore


def fuse_parallel_linears(gm: GraphModule, keep_weights: bool = False) -> GraphModule:
    """If there are parallel linears in the model, fuse them together.

    .. rubric:: Example

    .. testsetup::

        import torch
        import torch.nn as nn
        from torch.fx import symbolic_trace
        from composer.utils.fx_utils import count_op_instances, fuse_parallel_linears

    .. doctest::

        >>> class M(nn.Module):
        ...   def __init__(self):
        ...     super().__init__()
        ...     self.fc1 = nn.Linear(64, 64)
        ...     self.fc2 = nn.Linear(64, 64)
        ...   def forward(self, x):
        ...     y = self.fc1(x)
        ...     z = self.fc2(x)
        ...     return y + z
        >>> module = M()
        >>> traced = symbolic_trace(module)
        >>> count_op_instances(traced, nn.Linear)
        2
        >>> gm = fuse_parallel_linears(traced)
        >>> count_op_instances(traced, nn.Linear)
        1

    Arguments:
        gm (GraphModule): The source FX-traced graph.

    Returns:
        GraphModule: Modified GraphModule with parallel linears fused.
    """
    all_modules: Dict[str, nn.Module] = dict(gm.named_modules())
    fused_count = 0
    for node in gm.graph.nodes:
        # There could be more than two parallel linears
        linears_to_fuse = []

        # Check all the users of current node and collect all linear layers
        for user in list(node.users):
            if user.op == 'call_module' and isinstance(all_modules[user.target], nn.Linear):
                linears_to_fuse.append(user)

        # Fuse if there are more than 1 parallel linear layers
        if len(linears_to_fuse) > 1 and _can_linears_be_fused(linears_to_fuse, all_modules):
            lin, out_features = _create_fused_linear(linears_to_fuse, all_modules, keep_weights)
            gm.add_submodule(f'fused_linear_{fused_count}', lin)  # type: ignore
            with gm.graph.inserting_after(node):
                fused_node = gm.graph.call_module(f'fused_linear_{fused_count}', args=(node,))
            # insert the split node
            with gm.graph.inserting_after(fused_node):
                kwargs = {'split_size_or_sections': out_features, 'dim': -1}
                split_node = gm.graph.call_function(torch.split, args=(fused_node,), kwargs=kwargs)

            insert_point = split_node
            for idx, lin_node in enumerate(linears_to_fuse):
                with gm.graph.inserting_after(insert_point):
                    split_item = gm.graph.call_function(operator.getitem, (split_node, idx), {})
                lin_node.replace_all_uses_with(split_item)
                insert_point = split_item
                gm.graph.erase_node(lin_node)
            fused_count += 1
            gm.graph.lint()

    if fused_count > 0:
        gm.recompile()
    return gm
