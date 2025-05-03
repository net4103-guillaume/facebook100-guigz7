import networkx as nx
from typing import List, Callable, Any
import copy


Node = str
Value = int


def list_attribute(attribute: str, graph: nx.Graph) -> (List[Node], List[int]):
    """
    List all the existing values for a given attribute. Also gives the number of occurrences.

    :param attribute:
    :param graph:
    :return:
    """
    k = []
    v = []
    for node in graph:
        a = graph.nodes[node][attribute]
        if a not in k:
            k.append(a)
            v.append(1)
        else:
            v[k.index(a)] += 1
    return k, v


def print_list_attribute(attribute: str, graph: nx.Graph) -> None:
    k, v = list_attribute(attribute, graph)
    for x, y in zip(k, v):
        print(f"Year: {x}, size: {y}")


def get_subgraph_by_attribute(attribute: str, values: List[Value], graph: nx.Graph) -> nx.Graph:
    """
    Create a subgraph with all the nodes that satisfy the condition: node[attribute] in values.

    :param attribute:
    :param values:
    :param graph:
    :return:
    """
    g = copy.deepcopy(graph)
    for node in graph:
        if graph.nodes[node][attribute] not in values:
            g.remove_node(node)
    return g


def newman_girvan(g: nx.Graph, n_communities: int = 2) -> List[List[Node]]:
    """
    Newman Girvan's algorithm implementation.

    :param g:
    :param n_communities:
    :return:
    """
    max_iter = nx.number_of_edges(g)
    for _ in range(max_iter):
        edge_bw = nx.edge_betweenness_centrality(g)
        edge_to_remove = sorted(edge_bw.items(), key=lambda x: x[1])[-1][0]
        g.remove_edge(edge_to_remove[0], edge_to_remove[1])
        if nx.number_connected_components(g) >= n_communities:
            s = [list(g.subgraph(c).copy()) for c in nx.connected_components(g)]
            return s


def get_bisection_stat(
        graph: nx.Graph,
        attribute: str,
        values: List[Value],
        algorithm: Callable = nx.community.kernighan_lin_bisection
) -> List[Any]:
    """
    Get community information about two different populations.

    :param graph:
    :param attribute:
    :param values:
    :param algorithm:
    :return:
    """
    subgraph = get_subgraph_by_attribute(attribute, values, graph)
    communities = algorithm(subgraph)
    modularity = nx.community.modularity(subgraph, communities)
    partition_quality = nx.community.partition_quality(subgraph, communities)
    r = []
    for c in communities:
        k1, k2 = 0, 0
        for node in c:
            if subgraph.nodes[node][attribute] == values[0]:
                k1 += 1
            else:
                k2 += 1
        r.append((
            k1+k2,                          # Number of student in the partition
            round(k1/(k1+k2)*100),          # Percentage of population 1
            round(k2/(k1+k2)*100),          # Percentage of population 2
            round(modularity, 2),           # Modularity
            (round(partition_quality[0], 2), round(partition_quality[1], 2))  # Partition quality
        ))
    return r
