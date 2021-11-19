from typing import Dict

import networkx as nx
import numpy as np
import torch
import torch.fx
from torch.fx.node import Node


def parse_graph(model, x):
    G = nx.MultiDiGraph()

    trace = torch.fx.symbolic_trace(model)
    modules = dict(trace.named_modules())
    #import pdb; pdb.set_trace()
    ShapeProp(trace).propagate(x)

    for i, torch_node in enumerate(trace.graph.nodes):
        if torch_node.op == 'placeholder':
            G.add_node(0, cost=np.prod(torch_node.shape))
            G.add_edge(0, 1, cost=0, module=modules[torch_node._next.target])
            continue

        G.add_node(i, cost=np.prod(torch_node.shape))

        if torch_node.op == 'output':
            continue
        else:
            if torch_node._next.op in ['output']:
                module = torch.nn.Identity()
            else:
                module = modules[torch_node._next.target]
        G.add_edge(i, i + 1, cost=0, module=module)

    return G, 0, i


class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                import pdb; pdb.set_trace()
                
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result
