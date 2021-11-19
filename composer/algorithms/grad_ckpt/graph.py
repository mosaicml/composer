"""
This module implements basic functionality to operate on the graphs created.

The vertices represent the intermediate tensors and the edges represent DNN operations,
such as convolution, matrix multiplication, etc.

GC is formulated as an optimization problem, and it breaks the computation graph
into different segments, where we can perform re-forward and backward
independently. The algorithm divides the graph into Independent Segments
(IS) that form a division tree.

We can show that finding the optimal solution in the division tree is equivalent
 to finding the optimal solution in computation graph.

With the division tree, we can search for optimal GC
recursively. The recursion starts at the biggest IS (the whole
computation graph, root node of division tree) and ends at
the smallest IS (single vertex).
"""

from queue import Queue

import networkx as nx
import torch
from torch import nn as nn
from torch.utils.checkpoint import checkpoint


def __tuple_to_dict(t):
    lst = list(t)
    num = len(lst) // 3
    d = {}
    for i in range(num):
        tensor, s, ind = t[i * 3], t[i * 3 + 1], t[i * 3 + 2]
        d[(int(s), int(ind))] = tensor
    return d


def __dict_to_tuple(d):
    lst = []
    for (s, ind) in d:
        tensor = d[(s, ind)]
        lst.append(tensor)
        # has to use float otherwise throw requires_grad error
        lst.append(torch.tensor([float(s)], requires_grad=True))
        lst.append(torch.tensor([float(ind)], requires_grad=True))
    return tuple(lst)


def set_segment_training(segment, train=True):
    set_graph_training(segment.G, train=train)


def set_graph_training(graph, train=True):
    for e in graph.edges:
        module = graph.edges[e]['module']
        if isinstance(module, Segment):
            set_graph_training(module.G, train=train)
        else:
            if train:
                graph.edges[e]['module'].train()
            else:
                graph.edges[e]['module'].eval()


def replace_subgraph(graph1, graph2, source, target, id) -> nx.DiGraph:
    """ Function to replace subgraph in graph1 with graph2
    Args:
        graph1: networkx DiGraph
        graph2: networkx DiGraph
        source: source vertex in graph1
        target: target vertex in graph1
        id: if None, meaning source and target is not connected, else specify the connection id

    """
    if source not in graph1.nodes or target not in graph1.nodes:
        raise ValueError
    if id is None:
        nodes1 = set(nx.ancestors(graph1, target))
        nodes2 = set(nx.descendants(graph1, source))
        nodes = (nodes1.intersection(nodes2)).union(set({source, target}))
        edges_add_back = {}
        for node in nodes:
            for p in graph1.predecessors(node):
                if p not in nodes:
                    es = graph1.get_edge_data(p, node)
                    if es is not None:
                        for e in es:
                            edges_add_back[(p, node, e)] = es[e]
            for s in graph1.successors(node):
                if s not in nodes:
                    es = graph1.get_edge_data(node, s)
                    if es is not None:
                        for e in es:
                            edges_add_back[(node, s, e)] = es[e]
        for node in nodes:
            graph1.remove_node(node)
        for node in graph2.nodes:
            graph1.add_nodes_from({node: graph2.nodes[node]}, **graph2.nodes[node])
        for edge in graph2.edges:
            graph1.add_edges_from({edge: graph2.edges[edge]}, **graph2.edges[edge])
        for edge in edges_add_back:
            if edge not in graph1.edges:
                graph1.add_edges_from({edge: edges_add_back[edge]}, **edges_add_back[edge])
        return graph1
    else:
        graph1.remove_edge(source, target, id)
        for node in graph2.nodes:
            if node != source and node != target:
                graph1.add_nodes_from({node: graph2.nodes[node]}, **graph2.nodes[node])
        for edge in graph2.edges:
            graph1.add_edges_from({edge: graph2.edges[edge]}, **graph2.edges[edge])
        return graph1


def segment_checkpoint_forward(segment):

    def custom_forward(*inputs):
        outputs = segment(*inputs)
        return outputs

    return custom_forward


# NOTE: checkpoint autograd.function doesn't allow dictionary output, so have to use tensor to hold vertex id


def graph_forward(x,
                  G=None,
                  source=None,
                  target=None,
                  successors_dict=None,
                  predecessors_dict=None,
                  edges_dict=None,
                  do_checkpoint=True):
    """ Do checkpoint forward with each vertex in G as gradient checkpoint or do regular forward with G
    Args:
        G: networkx DAG
        source: source vertex key
        target: target vertex key
        x: input tensor
        do_checkpoint: whether to do regular forward or checkpoint forward

    """

    tensor_dict = {source: x}
    queue = Queue()
    queue.put(source)
    while not queue.empty():
        vertex_key = queue.get()
        for target_vertex_id in successors_dict[vertex_key]:
            edges = edges_dict[(vertex_key, target_vertex_id)]
            target_vertex = G.nodes[target_vertex_id]
            outputs = {}
            for id in edges:
                op = edges[id]['module']
                input = tensor_dict[vertex_key]
                if do_checkpoint:
                    output = checkpoint(segment_checkpoint_forward(op), input)
                else:
                    output = op(input)

                if type(output) == tuple:
                    output = __tuple_to_dict(output)
                    for key in output:
                        outputs[key] = output[key]
                else:
                    outputs[(vertex_key, id)] = output

            transition = target_vertex.get('transition', None)
            if transition is None:
                tensor_dict[target_vertex_id] = outputs[list(outputs.keys())[0]]
                queue.put(target_vertex_id)
            else:
                # handle multi inputs
                transition_input_order = target_vertex['transition_input_order']
                num_input = len(transition_input_order)

                inputs_for_transit = tensor_dict.get(target_vertex_id, {})
                for key in outputs:
                    inputs_for_transit[key] = outputs[key]
                if len(inputs_for_transit) == num_input:
                    inputs = [inputs_for_transit[i] for i in transition_input_order]
                    tensor_dict[target_vertex_id] = transition(inputs)
                    queue.put(target_vertex_id)
                else:
                    tensor_dict[target_vertex_id] = inputs_for_transit
    if type(tensor_dict[target]) == dict:
        return __dict_to_tuple(tensor_dict[target])
    else:
        return tensor_dict[target]


class Segment(nn.Module):
    """
    Wrapper class for inference with DAG.
    """

    def __init__(self, G, source, target, do_checkpoint=False):
        super(Segment, self).__init__()
        self.G = G
        self.source = source
        self.target = target
        self.info_dict = self.prepare_for_forward(G, source, target, do_checkpoint)

    def prepare_for_forward(self, G, source, target, do_checkpoint):
        info_dict = {'G': G, 'source': source, 'target': target}
        successors_dict, predecessors_dict, edges_dict = {}, {}, {}
        for v in G.nodes:
            predecessors_dict[v] = [n for n in G.predecessors(v)]
            successors_dict[v] = [n for n in G.successors(v)]
        for key in G.edges:
            e = G.edges[key]
            start, end, id = key
            if (start, end) not in edges_dict:
                edges_dict[(start, end)] = {}
            edges_dict[(start, end)][id] = e
        info_dict.update(successors_dict=successors_dict,
                         predecessors_dict=predecessors_dict,
                         edges_dict=edges_dict,
                         do_checkpoint=do_checkpoint)
        return info_dict

    def forward(self, x):
        return graph_forward(x, **self.info_dict)
