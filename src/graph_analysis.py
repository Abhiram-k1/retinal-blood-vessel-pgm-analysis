"""
Graph-Theoretic Analysis Module.
Calculates topological network properties: node count, edge count, degree distribution,
connected components, and network density.
"""

import networkx as nx
import numpy as np


def analyze_graph_topology(G: nx.Graph) -> dict:
    """
    Computes graph-theoretic metrics on vessel graph G=(V,E).
    
    Returns:
        dict containing:
            - 'nodes': total node count N = |V|
            - 'edges': total edge count E = |E|
            - 'connected_components': number of disjoint components
            - 'network_density': D = 2E / [N(N-1)]
            - 'mean_degree': average degree of nodes
            - 'max_degree': highest degree observed
            - 'branch_count': number of nodes with degree >= 3
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()
    
    if N == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "connected_components": 0,
            "network_density": 0.0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "branch_count": 0
        }
        
    num_components = nx.number_connected_components(G)
    
    # Network density D = 2E / [N(N-1)]
    density = (2.0 * E) / (N * (N - 1)) if N > 1 else 0.0
    
    degrees = [deg for _, deg in G.degree()]
    mean_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
    max_deg = int(np.max(degrees)) if len(degrees) > 0 else 0
    branch_count = int(np.sum(np.array(degrees) >= 3))
    
    return {
        "nodes": int(N),
        "edges": int(E),
        "connected_components": int(num_components),
        "network_density": float(density),
        "mean_degree": float(mean_deg),
        "max_degree": int(max_deg),
        "branch_count": int(branch_count)
    }
