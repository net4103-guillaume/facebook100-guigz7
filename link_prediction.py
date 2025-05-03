from abc import ABC, abstractmethod
from itertools import combinations
from math import log
from typing import List, Tuple
import networkx as nx


class LinkPrediction(ABC):
    def __init__(self, graph: nx.Graph):
        """
        Constructor

        :param graph:
        """
        self.graph = graph
        self.N = len(graph)

    def neighbors(self, v: int) -> List[Tuple[int, int]]:
        """
        Returns the neighbors list of a node

        :param v:
        :return:
        """
        return list(self.graph.neighbors(v))

    @abstractmethod
    def fit(self):
        raise NotImplementedError("Fit must be implemented")


class CommonNeighbors(LinkPrediction):
    def __init__(self, graph: nx.Graph):
        super(CommonNeighbors, self).__init__(graph)

    def fit(self):
        pass

    def predict(self, node_pairs: List[Tuple[int, int]]) -> List[int]:
        return [len(set(self.neighbors(u)).intersection(self.neighbors(v))) for u, v in node_pairs]


class JaccardCoefficient(LinkPrediction):
    def __init__(self, graph: nx.Graph):
        super(JaccardCoefficient, self).__init__(graph)

    def fit(self):
        pass

    def predict(self, node_pairs: List[Tuple[int, int]]) -> List[float]:
        scores = []
        for u, v in node_pairs:
            neighbors_u = set(self.neighbors(u))
            neighbors_v = set(self.neighbors(v))
            union_uv = neighbors_u.union(neighbors_v)
            intersection_uv = neighbors_u.intersection(neighbors_v)
            score = len(intersection_uv) / len(union_uv) if union_uv else 0
            scores.append(score)
        return scores


class AdamicAdar(LinkPrediction):
    def fit(self):
        pass

    def predict(self, node_pairs: List[Tuple[int, int]]) -> List[float]:
        scores = []
        for u, v in node_pairs:
            common_neighbors = set(self.neighbors(u)).intersection(self.neighbors(v))
            score = sum(1 / log(len(self.neighbors(w))) for w in common_neighbors if len(self.neighbors(w)) > 1)
            scores.append(score)
        return scores


def generate_non_edges(graph: nx.Graph) -> List[Tuple[int, int]]:
    """
    Generate non-edges node pairs for the graph

    :param graph:
    :return:
    """
    all_nodes = list(graph.nodes)
    potential_pairs = combinations(all_nodes, 2)
    non_edges = [pair for pair in potential_pairs if not graph.has_edge(*pair)]
    return non_edges
