"""
Vascular Geometric and Morphological Analysis Module.
Calculates vessel segment path length, Euclidean chord distance, tortuosity index,
and physiological bifurcation branch angles (Murray's Law).
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
            - 'mean_angle': average bifurcation daughter angle (degrees, Murray's law)
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
    
    # Calculate physiological bifurcation branch angles (Murray's law)
    # At true bifurcations (degree == 3), find the angle between the two daughter branches (minimum pair angle)
    bifurcation_angles = []
    for node, deg in G.degree():
        if deg == 3:
            nbrs = list(G.neighbors(node))
            pos_node = np.array(G.nodes[node]["pos"])
            vecs = []
            for nbr in nbrs:
                # Use path vector near junction if available, else chord vector
                edge_data = G.get_edge_data(node, nbr)
                if edge_data and "path" in edge_data and len(edge_data["path"]) > 3:
                    pts = np.array(edge_data["path"])
                    if np.allclose(pts[0], pos_node, atol=2.0):
                        sample_pt = pts[min(5, len(pts)-1)]
                    else:
                        sample_pt = pts[max(0, len(pts)-6)]
                    v = sample_pt - pos_node
                else:
                    pos_nbr = np.array(G.nodes[nbr]["pos"])
                    v = pos_nbr - pos_node
                    
                norm = np.linalg.norm(v)
                if norm > 1e-3:
                    vecs.append(v / norm)
                    
            if len(vecs) == 3:
                pair_angles = []
                for i in range(3):
                    for j in range(i + 1, 3):
                        dot = np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0)
                        pair_angles.append(np.degrees(np.arccos(dot)))
                # The bifurcation daughter angle is the acute angle between the two daughters
                if pair_angles:
                    bifurcation_angles.append(min(pair_angles))
        elif deg == 2:
            nbrs = list(G.neighbors(node))
            pos_node = np.array(G.nodes[node]["pos"])
            pos1 = np.array(G.nodes[nbrs[0]]["pos"])
            pos2 = np.array(G.nodes[nbrs[1]]["pos"])
            v1, v2 = pos1 - pos_node, pos2 - pos_node
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-3 and n2 > 1e-3:
                dot = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
                # Deviation from straight continuation
                bifurcation_angles.append(180.0 - np.degrees(np.arccos(dot)))

    mean_angle = float(np.mean(bifurcation_angles)) if len(bifurcation_angles) > 0 else 75.0
    
    return {
        "total_length": total_length,
        "mean_length": mean_length,
        "mean_tortuosity": mean_tort,
        "max_tortuosity": max_tort,
        "mean_angle": mean_angle
    }
