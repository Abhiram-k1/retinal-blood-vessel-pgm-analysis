"""
Global Network Efficiency Module for Retinal Vascular Networks.
Measures the capacity for parallel information/fluid transport across the network,
strictly handling disconnected components without arbitrary penalties.
"""

import networkx as nx
import numpy as np


def compute_global_network_efficiency(G: nx.Graph, weight: str = "length") -> float:
    """
    Computes global efficiency E_global of vessel graph G.
    
    E_global = 1 / [N(N-1)] * sum_{i != j} 1 / d(i, j)
    
    If i and j are disconnected, d(i, j) = infinity -> 1 / d(i, j) = 0.
    """
    N = G.number_of_nodes()
    if N <= 1:
        return 0.0
        
    inv_dist_sum = 0.0
    
    # Calculate shortest path lengths for all pairs within connected components
    for comp in nx.connected_components(G):
        if len(comp) <= 1:
            continue
        subg = G.subgraph(comp)
        # All pairs shortest path lengths
        lengths = dict(nx.all_pairs_dijkstra_path_length(subg, weight=weight))
        for u, dist_dict in lengths.items():
            for v, d in dist_dict.items():
                if u != v and d > 1e-4:
                    inv_dist_sum += (1.0 / d)
                    
    normalization = N * (N - 1)
    e_global = float(inv_dist_sum / normalization)
    return e_global
