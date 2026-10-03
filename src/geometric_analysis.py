"""
Vascular Geometric and Morphological Analysis Module.
Calculates vessel segment path length, Euclidean chord distance, tortuosity index,
and bifurcation branch angles.
"""

import numpy as np
import networkx as nx


def analyze_vessel_geometry(G: nx.Graph) -> dict:
    """
    Computes geometric metrics across all edges and junctions in vessel graph G.
    
    Returns:
        dict containing:
            - 'total_length': sum of all edge curve lengths
            - 'mean_length': average edge curve length
            - 'mean_tortuosity': average tortuosity index across all edges (L / d)
            - 'max_tortuosity': maximum tortuosity observed
            - 'mean_angle': average bifurcation angle (degrees)
    """
    if G.number_of_edges() == 0:
        return {
            "total_length": 0.0,
            "mean_length": 0.0,
            "mean_tortuosity": 1.0,
            "max_tortuosity": 1.0,
            "mean_angle": 0.0
        }
        
    lengths = []
    tortuosities = []
    
    for u, v, data in G.edges(data=True):
        L = float(data.get("length", 1.0))
        d = float(data.get("euclidean_distance", 1.0))
        lengths.append(L)
        
        # Division by zero guard
        if d > 1e-2:
            t = max(1.0, L / d)
        else:
            t = 1.0
        tortuosities.append(t)
        
    total_length = float(np.sum(lengths))
    mean_length = float(np.mean(lengths)) if len(lengths) > 0 else 0.0
    mean_tort = float(np.mean(tortuosities)) if len(tortuosities) > 0 else 1.0
    max_tort = float(np.max(tortuosities)) if len(tortuosities) > 0 else 1.0
    
    # Calculate bifurcation branch angles at junction nodes (degree >= 2)
    angles = []
    for node, deg in G.degree():
        if deg >= 2:
            nbrs = list(G.neighbors(node))
            pos_node = np.array(G.nodes[node]["pos"])
            # Compute vectors from node to each neighbor
            vecs = []
            for nbr in nbrs:
                pos_nbr = np.array(G.nodes[nbr]["pos"])
                v = pos_nbr - pos_node
                norm = np.linalg.norm(v)
                if norm > 1e-3:
                    vecs.append(v / norm)
                    
            # Compute pairwise angles between branch vectors
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    dot = np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(dot))
                    angles.append(angle_deg)
                    
    mean_angle = float(np.mean(angles)) if len(angles) > 0 else 0.0
    
    return {
        "total_length": total_length,
        "mean_length": mean_length,
        "mean_tortuosity": mean_tort,
        "max_tortuosity": max_tort,
        "mean_angle": mean_angle
    }
