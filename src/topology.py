from __future__ import annotations

import networkx as nx
import numpy as np


FALCON_27_EDGES = [
    (0, 1),
    (1, 2),
    (1, 4),
    (2, 3),
    (3, 5),
    (4, 7),
    (5, 8),
    (6, 7),
    (7, 10),
    (8, 11),
    (9, 10),
    (10, 12),
    (11, 14),
    (12, 13),
    (12, 15),
    (13, 14),
    (14, 16),
    (15, 18),
    (16, 19),
    (17, 18),
    (18, 21),
    (19, 22),
    (20, 21),
    (21, 23),
    (22, 25),
    (23, 24),
    (24, 25),
    (25, 26),
]


def build_falcon_27_graph() -> nx.Graph:
    """Return a NetworkX undirected graph for IBM Falcon 27q heavy-hex topology."""
    graph = nx.Graph()
    graph.add_nodes_from(range(27))
    graph.add_edges_from(FALCON_27_EDGES)
    return graph


def compute_distance_matrix(graph: nx.Graph) -> np.ndarray:
    """Return all-pairs shortest-path distances as a NumPy array."""
    return np.asarray(nx.floyd_warshall_numpy(graph), dtype=np.float32)
