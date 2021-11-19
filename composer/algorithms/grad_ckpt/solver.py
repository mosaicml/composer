# Code adapted from https://github.com/lordfjw/OptimalGradCheckpointing
# Paper at https://arxiv.org/abs/1808.00079, Optimal Gradient Checkpoint search for arbitrary computation graphs


import numpy as np
import networkx as nx
from queue import Queue
from tqdm import tqdm
from composer.algorithms.grad_ckpt.utils import add_vertex_cost_to_edge
from composer.algorithms.grad_ckpt.graph import replace_subgraph, Segment


class ArbitrarySolver():
    def __init__(self):
        self.linear_solver = LinearSolver()

    def solve(self, G, source, target, use_tqdm=True):
        print('Building Division Tree')
        graph, cost, division_type = self.build_division_tree(
            G, source, target)
        print('Getting Max Terms')
        max_terms = self.get_max_terms(graph, division_type, max_terms=set())

        best_run_graph = None
        best_total_cost = np.inf
        max_terms = list(set(max_terms))
        if 0 in max_terms:
            max_terms.remove(0)
        max_terms = sorted(max_terms, reverse=True)
        best_max_term = -1
        best_gc_cost = -1
        print('Solving Optimal for Each Max Term')
        if use_tqdm:
            iterator = tqdm(max_terms)
        else:
            iterator = max_terms
        for max_term in iterator:
            run_graph, gc_cost = self.solve_with_max(
                graph, source, target, max_term, division_type)
            total_cost = gc_cost + max_term
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_run_graph = run_graph
                best_max_term = max_term
                best_gc_cost = gc_cost

        best_run_graph = self.optimize_run_graph(G, best_run_graph)
        return best_run_graph, best_total_cost

    def optimize_run_graph(self, G, run_graph):
        # flatten run graph to avoid too many recursions, and other optimization to speed up

        new_run_graph = run_graph.copy()

        for edge_key in run_graph.edges:
            source, target, id = edge_key
            edge_nodes, edge_edges = self.nodes_edges_in_run_graph_edge(
                run_graph, edge_key)
            subgraph = nx.MultiDiGraph()
            for sub_node_key in edge_nodes:
                subgraph.add_node(sub_node_key, **G.nodes[sub_node_key])
            for sub_edge_key in edge_edges:
                subgraph.add_edges_from(
                    {sub_edge_key: G.edges[sub_edge_key]}, **G.edges[sub_edge_key])
            new_run_graph.edges[(source, target, id)]['graph'] = subgraph
            new_run_graph.edges[(source, target, id)]['module'] = Segment(
                subgraph, source, target)
            if 'transition' in new_run_graph.nodes[target]:
                trans_complete = True
                for (trans_source, trans_id) in new_run_graph.nodes[target]['transition_input_order']:
                    if trans_source not in edge_nodes:
                        trans_complete = False
                        break
                if trans_complete:
                    # transition handled in subgraph
                    new_run_graph.nodes[target]['transition'] = None
                    new_run_graph.nodes[target]['transition_input_order'] = None

        return new_run_graph

    def nodes_edges_in_run_graph_edge(self, graph, edge_key):
        start, end, id = edge_key

        nodes = set()
        nodes.add(start)
        nodes.add(end)
        edges = dict()

        edge = graph.edges[edge_key]
        subgraph = edge['module']
        if type(subgraph) == Segment:
            subgraph = subgraph.G
            for e in subgraph.edges:
                subnodes, subedges = self.nodes_edges_in_run_graph_edge(
                    subgraph, e)
                nodes = nodes.union(subnodes)
                for key in subedges:
                    edges[key] = subedges[key]
            return nodes, edges
        else:
            edges[edge_key] = edge
            return nodes, edges

    def edge_in_graph(self, run_graph, edge):
        if edge in run_graph.edges:
            return True
        flag = False
        for e in run_graph.edges:
            subgraph = run_graph.edges[e]['module']
            if type(subgraph) == Segment:
                flag = flag or self.edge_in_graph(subgraph.G, edge)
            if flag:
                return flag
        return flag

    def solve_with_max(self, division_tree, source, target, max_term, division_type):
        run_graph = division_tree.copy()
        total_cost = 0
        if division_type != 'linear':
            for (s, t, id) in division_tree.edges:
                cost = division_tree.edges[(s, t, id)]['cost']
                subtree = division_tree.edges[(s, t, id)]['graph']
                if cost > max_term:
                    sub_run_graph, GC_cost = self.solve_with_max(
                        subtree, s, t, max_term, division_tree.edges[(s, t, id)]['division_type'])
                    run_graph = replace_subgraph(
                        run_graph, sub_run_graph, s, t, id)
                    total_cost += GC_cost
            for node in division_tree.nodes:
                if node != source and node != target:
                    total_cost += division_tree.nodes[node]['cost']
            return run_graph, total_cost
        else:
            nodes = list(nx.topological_sort(division_tree))
            linear_subgraph_vertices = [source]
            GC_set = set()
            for i in range(len(nodes) - 1):
                node1, node2 = nodes[i], nodes[i+1]
                edge = division_tree.edges[(node1, node2, 0)]
                if edge['cost'] > max_term:
                    sub_run_graph, GC_cost = self.solve_with_max(
                        edge['graph'], node1, node2, max_term, edge['division_type'])
                    run_graph = replace_subgraph(
                        run_graph, sub_run_graph, node1, node2, 0)

                    total_cost += GC_cost
                    GC_set.add(node1)
                    GC_set.add(node2)
                    if len(linear_subgraph_vertices) > 1:
                        linear_subgraph = division_tree.subgraph(
                            linear_subgraph_vertices).copy()
                        GCPs = self.solve_linear(linear_subgraph, max_term)
                        GC_set = GC_set.union(set(GCPs))
                        start_node = GCPs[0]
                        linear_run_graph = nx.MultiDiGraph()
                        linear_run_graph.add_nodes_from(
                            {start_node: linear_subgraph.nodes[start_node]}, **linear_subgraph.nodes[start_node])
                        for j in range(1, len(GCPs)):
                            end_node = GCPs[j]
                            linear_run_graph.add_nodes_from({end_node: linear_subgraph.nodes[end_node]},
                                                            **linear_subgraph.nodes[end_node])
                            start_index = nodes.index(start_node)
                            end_index = nodes.index(end_node)
                            edge_subgraph = linear_subgraph.subgraph(
                                nodes[start_index:(end_index+1)]).copy()
                            linear_run_graph.add_edge(start_node, end_node, module=Segment(
                                edge_subgraph, start_node, end_node))
                            start_node = end_node

                        run_graph = replace_subgraph(
                            run_graph, linear_run_graph, linear_subgraph_vertices[0], linear_subgraph_vertices[-1], id=None)
                        linear_subgraph_vertices = [node2]
                    else:
                        linear_subgraph_vertices = [node2]
                else:
                    linear_subgraph_vertices.append(node2)

            if len(linear_subgraph_vertices) > 1:
                linear_subgraph = division_tree.subgraph(
                    linear_subgraph_vertices).copy()
                GCPs = self.solve_linear(linear_subgraph, max_term)
                GC_set = GC_set.union(set(GCPs))

                start_node = GCPs[0]
                linear_run_graph = nx.MultiDiGraph()
                linear_run_graph.add_nodes_from({start_node: linear_subgraph.nodes[start_node]},
                                                **linear_subgraph.nodes[start_node])
                for j in range(1, len(GCPs)):
                    end_node = GCPs[j]
                    linear_run_graph.add_nodes_from({end_node: linear_subgraph.nodes[end_node]},
                                                    **linear_subgraph.nodes[end_node])
                    start_index = nodes.index(start_node)
                    end_index = nodes.index(end_node)
                    edge_subgraph = linear_subgraph.subgraph(
                        nodes[start_index:(end_index+1)]).copy()
                    linear_run_graph.add_edge(start_node, end_node, module=Segment(
                        edge_subgraph, start_node, end_node))
                    start_node = end_node
                run_graph = replace_subgraph(
                    run_graph, linear_run_graph, linear_subgraph_vertices[0], linear_subgraph_vertices[-1], id=None)

            for node in GC_set:
                if node != source and node != target:
                    total_cost += division_tree.nodes[node]['cost']
            return run_graph, total_cost

    def solve_linear(self, linear_subgraph, max_term):
        linear_node_num = len(linear_subgraph.nodes)
        linear_nodes = list(nx.topological_sort(linear_subgraph))
        AG = linear_subgraph.copy()

        edge_costs = np.zeros([linear_node_num, linear_node_num])
        cost_sums = np.zeros([linear_node_num])
        cost_sums[0] = linear_subgraph.nodes[linear_nodes[0]]['cost']
        for j in range(1, linear_node_num):
            cost_sums[j] = cost_sums[j - 1] + linear_subgraph.nodes[linear_nodes[j]]['cost'] + \
                linear_subgraph.edges[(
                    linear_nodes[j - 1], linear_nodes[j], 0)]['cost']

        for j in range(linear_node_num - 1):
            for k in range(j + 1, linear_node_num):
                edge_cost = cost_sums[k] - cost_sums[j] - \
                    linear_subgraph.nodes[linear_nodes[k]]['cost']
                edge_costs[j, k] = edge_cost

        for j in range(linear_node_num):
            for k in range(j + 1, linear_node_num):
                if edge_costs[j, k] <= max_term:
                    # add accessibility edge
                    AG.add_edge(linear_nodes[j], linear_nodes[k], cost=0)
        # move vertex cost to edge cost to use networkx shortest path algorithm
        AG = add_vertex_cost_to_edge(AG)
        GCPs = nx.shortest_path(
            AG, source=linear_nodes[0], target=linear_nodes[-1], weight='weight')
        return GCPs

    def get_max_terms(self, division_tree, division_type, max_terms=set()):
        if division_type == 'linear':
            nodes = [n for n in nx.topological_sort(division_tree)]
            node_num = len(nodes)

            cost_sums = np.zeros([node_num])
            cost_sums[0] = division_tree.nodes[nodes[0]]['cost']
            for j in range(1, node_num):
                cost_sums[j] = cost_sums[j - 1] + division_tree.nodes[nodes[j]]['cost'] + \
                    division_tree.edges[(nodes[j - 1], nodes[j], 0)]['cost']

            for j in range(node_num - 1):
                for k in range(j + 1, node_num):
                    edge_cost = cost_sums[k] - cost_sums[j] - \
                        division_tree.nodes[nodes[k]]['cost']
                    max_terms.add(edge_cost)

            for (s, t, id) in division_tree.edges:
                max_terms.add(division_tree.edges[(s, t, id)]['cost'])
                graph = division_tree.edges[(s, t, id)].get('graph', None)
                divi_type = division_tree.edges[(s, t, id)].get(
                    'division_type', None)
                if graph is not None:
                    max_terms = self.get_max_terms(graph, divi_type, max_terms)
            return max_terms

        elif division_type == 'leave':
            return max_terms
        else:
            for (s, t, id) in division_tree.edges:
                max_terms.add(division_tree.edges[(s, t, id)]['cost'])
                graph = division_tree.edges[(s, t, id)].get('graph', None)
                divi_type = division_tree.edges[(s, t, id)].get(
                    'division_type', None)
                if graph is not None:
                    max_terms = self.get_max_terms(graph, divi_type, max_terms)
            return max_terms

    def build_division_tree(self, G, source, target, known_division_type=None):
        if len(G.nodes) == 2:
            return G, 0, 'leave'
        divisions, source_targets, division_type = self.get_division(
            G, source, target, division_type=None, known_division_type=known_division_type)
        if len(divisions) == 1:
            # leaves of division tree
            return G, 0, 'leave'
        division_tree = nx.MultiDiGraph()
        total_cost = 0
        for subgraph, source_target in zip(divisions, source_targets):
            s, t = source_target
            if s not in division_tree.nodes:
                division_tree.add_nodes_from({s: G.nodes[s]}, **G.nodes[s])
            if t not in division_tree.nodes:
                division_tree.add_nodes_from({t: G.nodes[t]}, **G.nodes[t])
            id = division_tree.add_edge(s, t)
            graph, cost, sub_division_type = self.build_division_tree(
                subgraph, s, t, known_division_type=division_type)
            division_tree.edges[(s, t, id)]['graph'] = graph
            division_tree.edges[(s, t, id)]['module'] = Segment(graph, s, t)
            division_tree.edges[(s, t, id)]['cost'] = cost
            division_tree.edges[(s, t, id)
                                ]['division_type'] = sub_division_type
            total_cost += cost
        for node in division_tree.nodes:
            if node != source and node != target:
                total_cost += division_tree.nodes[node]['cost']
        return division_tree, total_cost, division_type

    def get_division(self, G, source, target, division_type=None, known_division_type=None):
        if division_type is None:
            nodes = list(nx.topological_sort(G))
            nodes.remove(source)
            nodes.remove(target)

            if known_division_type == 'linear':
                # skipped finding splitting vertex
                pass
            else:
                for node in nodes:
                    ancestors = nx.ancestors(G, node)
                    descendants = nx.descendants(G, node)
                    subgraph1 = G.subgraph(list(ancestors) + [node])
                    subgraph2 = G.subgraph(list(descendants) + [node])
                    edges1 = set(subgraph1.edges)
                    edges2 = set(subgraph2.edges)
                    edges = set(G.edges)
                    if edges1.union(edges2) == edges and len(edges1.intersection(edges2)) == 0:
                        # found splitting vertex
                        division1, source_targets1 = [
                            subgraph1.copy()], [[source, node]]
                        division2, source_targets2, _ = self.get_division(
                            subgraph2.copy(), node, target, division_type='linear')
                        return division1 + division2, source_targets1 + source_targets2, 'linear'

            # no splitting vertex found, check whether it has branches
            source_target_edge = G.get_edge_data(source, target)
            if source_target_edge is not None:
                subgraph1 = G.subgraph([source, target]).copy()
                subgraph2 = G.copy()
                for id in G.get_edge_data(source, target):
                    subgraph2.remove_edge(source, target, id)
                division2, source_targets2, _ = self.get_division(
                    subgraph2, source, target, division_type='branch')
                return [subgraph1] + division2, [[source, target]] + source_targets2, 'branch'
            else:
                random_node = nodes[np.random.choice(len(nodes))]
                ancestors = nx.ancestors(G, random_node)
                descendants = nx.descendants(G, random_node)
                subgraph1_nodes = list(ancestors) + \
                    [random_node] + list(descendants)
                queue = Queue()
                for node in subgraph1_nodes:
                    if node != source and node != target:
                        for p in G.predecessors(node):
                            if p not in subgraph1_nodes:
                                queue.put(p)
                                subgraph1_nodes.append(p)
                        for s in G.successors(node):
                            if s not in subgraph1_nodes:
                                queue.put(s)
                                subgraph1_nodes.append(s)
                while not queue.empty():
                    n = queue.get()
                    for p in G.predecessors(n):
                        if p not in subgraph1_nodes:
                            queue.put(p)
                            subgraph1_nodes.append(p)
                    for s in G.successors(n):
                        if s not in subgraph1_nodes:
                            queue.put(s)
                            subgraph1_nodes.append(s)

                subgraph1 = G.subgraph(subgraph1_nodes).copy()
                edges1 = set(subgraph1.edges)
                edges = set(G.edges)
                if edges1 == edges:
                    # division type is complicate
                    return self.get_division(G, source, target, division_type='complicate')
                else:
                    subgraph2_nodes = set(nodes).difference(
                        set(subgraph1_nodes))
                    subgraph2_nodes = list(subgraph2_nodes) + [source, target]
                    subgraph2 = G.subgraph(subgraph2_nodes).copy()
                    division2, source_targets2, divi_type = self.get_division(
                        subgraph2, source, target, division_type='branch')
                    return [subgraph1] + division2, [[source, target]] + source_targets2, 'branch'

        elif division_type == 'linear':
            nodes = list(nx.topological_sort(G))
            if len(nodes) == 2:
                return [G], [[source, target]], division_type
            for node in nodes:
                if node == source or node == target:
                    continue
                ancestors = nx.ancestors(G, node)
                descendants = nx.descendants(G, node)
                subgraph1 = G.subgraph(list(ancestors) + [node])
                subgraph2 = G.subgraph(list(descendants) + [node])
                edges1 = set(subgraph1.edges)
                edges2 = set(subgraph2.edges)
                edges = set(G.edges)
                if edges1.union(edges2) == edges and len(edges1.intersection(edges2)) == 0:
                    # found splitting vertex
                    division1, source_targets1 = [
                        subgraph1.copy()], [[source, node]]
                    division2, source_targets2, divi_type = self.get_division(subgraph2.copy(), node, target,
                                                                              division_type='linear')
                    return division1 + division2, source_targets1 + source_targets2, division_type
            return [G], [[source, target]], division_type
        elif division_type == 'branch':
            nodes = list(nx.topological_sort(G))
            nodes.remove(source)
            nodes.remove(target)

            random_node = nodes[np.random.choice(len(nodes))]
            ancestors = nx.ancestors(G, random_node)
            descendants = nx.descendants(G, random_node)
            subgraph1_nodes = list(ancestors) + \
                [random_node] + list(descendants)
            queue = Queue()
            for node in subgraph1_nodes:
                if node != source and node != target:
                    for p in G.predecessors(node):
                        if p not in subgraph1_nodes:
                            queue.put(p)
                            subgraph1_nodes.append(p)
                    for s in G.successors(node):
                        if s not in subgraph1_nodes:
                            queue.put(s)
                            subgraph1_nodes.append(s)
            while not queue.empty():
                n = queue.get()
                for p in G.predecessors(n):
                    if p not in subgraph1_nodes:
                        queue.put(p)
                        subgraph1_nodes.append(p)
                for s in G.successors(n):
                    if s not in subgraph1_nodes:
                        queue.put(s)
                        subgraph1_nodes.append(s)
            subgraph1 = G.subgraph(subgraph1_nodes).copy()
            edges1 = set(subgraph1.edges)
            edges = set(G.edges)
            if edges1 == edges:
                # division type is complicate
                return [G], [[source, target]], division_type
            else:
                subgraph2_nodes = set(nodes).difference(set(subgraph1_nodes))
                subgraph2_nodes = list(subgraph2_nodes) + [source, target]
                subgraph2 = G.subgraph(subgraph2_nodes).copy()
                division2, source_targets2, divi_type = self.get_division(
                    subgraph2, source, target, division_type='branch')
                return [subgraph1] + division2, [[source, target]] + source_targets2, division_type
        elif division_type == 'complicate':
            nodes = list(nx.topological_sort(G))
            adjacency_matrix = np.array(
                nx.linalg.graphmatrix.adjacency_matrix(G, nodelist=nodes).todense())
            reverse_mapping = {node: i for i, node in enumerate(nodes)}
            # counting undirected adjacency
            adjacency_matrix = adjacency_matrix + adjacency_matrix.T
            path_adjacency_matrix = adjacency_matrix.copy()
            descendants_all = {node: nx.descendants(G, node) for node in nodes}
            ancestors_all = {node: nx.ancestors(G, node) for node in nodes}
            for node in nodes:
                idx = reverse_mapping[node]
                for n in descendants_all[node]:
                    path_adjacency_matrix[idx, reverse_mapping[n]] += 1
                for n in ancestors_all[node]:
                    path_adjacency_matrix[idx, reverse_mapping[n]] += 1

            candidate_nodes = [node for i, node in enumerate(nodes) if np.sum(
                adjacency_matrix[i, :]) > 2 or node == source or node == target]
            subgraphs = []
            source_targets_all = []
            for i in (range(len(candidate_nodes) - 1)):
                for j in range(len(candidate_nodes) - 1, i, -1):
                    node1, node2 = candidate_nodes[i], candidate_nodes[j]
                    if node1 == source and node2 == target:
                        continue

                    subgraph = self.get_largest_IS(G, node1, node2, descendants_all, ancestors_all,
                                                   adjacency_matrix, path_adjacency_matrix, np.array(nodes), reverse_mapping)

                    if subgraph is not None:
                        subgraphs.append(subgraph)
                        source_targets_all.append([node1, node2])
            subgraphs_source_targets = [[subgraph, source_target]
                                        for subgraph, source_target in zip(subgraphs, source_targets_all)]
            subgraphs_source_targets = sorted(
                subgraphs_source_targets, key=lambda x: -len(list(x[0].edges)))
            subgraphs = [subgraphs_source_target[0]
                         for subgraphs_source_target in subgraphs_source_targets]
            source_targets_all = [subgraphs_source_target[1]
                                  for subgraphs_source_target in subgraphs_source_targets]
            divisions = []
            source_targets = []
            occupied_edges = set()
            for subgraph, source_target in zip(subgraphs, source_targets_all):
                subgraph_edges = set(subgraph.edges)
                if len(subgraph_edges.intersection(occupied_edges)) == 0:
                    divisions.append(subgraph)
                    source_targets.append(source_target)
                    occupied_edges = occupied_edges.union(subgraph_edges)

            if len(occupied_edges) != len(G.edges):
                print('Something wrong with Complicate IS')
                raise ValueError

            return divisions, source_targets, division_type
        else:
            raise KeyError

    def get_largest_IS(self, G, node1, node2, descendants_all, ancestors_all, adjacency_matrix, path_adjacency_matrix, adjacency_nodes_mapping, reverse_mapping):

        ancestors = ancestors_all[node2]
        descendants = descendants_all[node1]
        subgraph_nodes = set(ancestors).intersection(set(descendants))
        subgraph_nodes = subgraph_nodes.union({node1, node2})
        other_nodes = set(G.nodes).difference(subgraph_nodes)

        subgraph_inside_nodes_idxs = np.array(
            [reverse_mapping[n] for n in subgraph_nodes.difference({node1, node2})])
        if len(subgraph_inside_nodes_idxs) == 0:
            subgraph = G.subgraph(subgraph_nodes)
        else:
            other_nodes_idxs = np.array(
                [reverse_mapping[n] for n in other_nodes])
            adjacency = np.sum(
                adjacency_matrix[other_nodes_idxs, :][:, subgraph_inside_nodes_idxs], axis=0)
            removed_inside_nodes_idxs = subgraph_inside_nodes_idxs[adjacency > 0]
            if len(removed_inside_nodes_idxs) > 0:

                prev_remove_num = len(removed_inside_nodes_idxs)
                while True:
                    remaining_subgraph_inside_nodes_idx = set(
                        subgraph_inside_nodes_idxs).difference(set(removed_inside_nodes_idxs))
                    if len(remaining_subgraph_inside_nodes_idx) <= 0:
                        break
                    remaining_subgraph_inside_nodes_idx = np.array(
                        list(remaining_subgraph_inside_nodes_idx))
                    path_adjacency_submat = path_adjacency_matrix[removed_inside_nodes_idxs,
                                                                  :][:, remaining_subgraph_inside_nodes_idx]
                    path_adjacency = np.sum(path_adjacency_submat, axis=0)
                    extra_removed_inside_nodes_idxs = remaining_subgraph_inside_nodes_idx[
                        path_adjacency > 0]
                    removed_inside_nodes_idxs = set(removed_inside_nodes_idxs).union(
                        set(extra_removed_inside_nodes_idxs))
                    cur_remove_num = len(removed_inside_nodes_idxs)
                    removed_inside_nodes_idxs = np.array(
                        list(removed_inside_nodes_idxs))
                    if cur_remove_num == prev_remove_num:
                        break
                    prev_remove_num = cur_remove_num
                removed_nodes = set(
                    adjacency_nodes_mapping[removed_inside_nodes_idxs])
                remaining_subgraph_nodes = subgraph_nodes.difference(
                    removed_nodes)
            else:
                remaining_subgraph_nodes = subgraph_nodes
            subgraph = G.subgraph(remaining_subgraph_nodes)

        if len(subgraph.edges) == 0 or (not nx.is_connected(subgraph.to_undirected())):
            return None
        else:
            return subgraph.copy()


class LinearSolver():
    def __init__(self):
        pass

    def solve(self, G, source, target):
        """ Solve optimal checkpoints for linear computation graph G, assuming G is linear DAG
        """
        node_num = len(G.nodes)
        max_terms, edge_costs, vertex_id_mapping = self.get_max_terms_and_costs(
            G, source)

        best_GCPs = None
        best_cost = np.inf
        for max_term in max_terms:
            GCPs, total_cost = self.solve_with_max(
                G, source, target, edge_costs, vertex_id_mapping, max_term)
            total_cost += max_term
            if total_cost < best_cost:
                best_GCPs = GCPs
                best_cost = total_cost

        run_graph = nx.MultiDiGraph()
        for i in range(len(best_GCPs) - 1):
            s = best_GCPs[i]
            t = best_GCPs[i + 1]
            subgraph_nodes = [s]
            vertex_id = s
            while vertex_id != t:
                vertex_id = [n for n in G.successors(vertex_id)][0]
                subgraph_nodes.append(vertex_id)
            subgraph = G.subgraph(subgraph_nodes).copy()
            if s not in run_graph.nodes:
                run_graph.add_node(s)
            if t not in run_graph.nodes:
                run_graph.add_node(t)
            run_graph.add_edge(s, t, module=Segment(subgraph, s, t))
        return run_graph, best_cost

    def solve_with_max(self, G, source, target, edge_costs, vertex_id_mapping, max_term):
        # construct accessibility graph
        node_num = len(G.nodes)
        AG = G.copy()
        for i in range(node_num):
            for j in range(i + 1, node_num):
                if edge_costs[i, j] <= max_term:
                    # add accessibility edge
                    AG.add_edge(
                        vertex_id_mapping[i], vertex_id_mapping[j], cost=0)
        # move vertex cost to edge cost to use networkx shortest path algorithm
        AG = add_vertex_cost_to_edge(AG)
        GCPs = nx.shortest_path(
            AG, source=source, target=target, weight='weight')
        total_cost = 0
        for vertex_id in GCPs:
            total_cost += G.nodes[vertex_id]['cost']

        return GCPs, total_cost

    def get_max_terms_and_costs(self, G, source):
        node_num = len(G.nodes)
        max_terms = set()
        cost_sums = np.zeros([node_num])
        edge_costs = np.zeros([node_num, node_num])
        vertex_id = source
        vertex_id_mapping = [source]
        cost_sums[0] = G.nodes[vertex_id]['cost']
        for i in range(1, node_num):
            vertex_id = [n for n in G.successors(vertex_id)][0]
            vertex_id_mapping.append(vertex_id)
            cost_sums[i] = cost_sums[i - 1] + G.nodes[vertex_id]['cost']

        for i in range(node_num - 1):
            for j in range(i + 2, node_num):
                cost = cost_sums[j - 1] - cost_sums[i]
                edge_costs[i, j] = cost
                max_terms.add(cost)

        return max_terms, edge_costs, vertex_id_mapping

