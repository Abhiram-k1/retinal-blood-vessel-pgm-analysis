"""
Mathematical Graph Construction Module.
Builds a NetworkX graph G=(V, E) from skeletonized vessels and detected nodes.
Nodes represent anatomical junctions/endpoints; edges represent vessel segments
with stored centerline paths, true curve lengths, and Euclidean chord distances.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image
from scipy.spatial import KDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "graphs"


def build_vessel_graph(skeleton: np.ndarray, nodes_df) -> nx.Graph:
    """
    Constructs a topological NetworkX Graph G=(V, E) from skeleton and node dataframe.
    
    Args:
        skeleton: uint8 or bool array (H, W), 1=skeleton, 0=background
        nodes_df: DataFrame with columns [node_id, x, y, node_type]
        
    Returns:
        G: networkx.Graph with node attributes ('pos', 'node_type') and edge attributes
           ('path', 'length', 'euclidean_distance', 'tortuosity')
    """
    G = nx.Graph()
    skel_b = skeleton > 0
    skel_y, skel_x = np.where(skel_b)
    
    if len(skel_x) == 0 or len(nodes_df) == 0:
        return G
        
    # 1. Add nodes to G
    node_coords = []
    node_ids = []
    for _, row in nodes_df.iterrows():
        nid = int(row["node_id"])
        x, y = float(row["x"]), float(row["y"])
        G.add_node(nid, pos=(x, y), node_type=row["node_type"])
        node_coords.append([x, y])
        node_ids.append(nid)
        
    node_tree = KDTree(node_coords)
    
    # 2. Build pixel-level graph from skeleton 8-connectivity
    # Map pixel (x, y) to integer index
    skel_points = np.column_stack([skel_x, skel_y])
    skel_tree = KDTree(skel_points)
    
    pixel_adj = nx.Graph()
    for i, (px, py) in enumerate(skel_points):
        pixel_adj.add_node(i, pos=(px, py))
        
    # Connect 8-neighbors (Euclidean distance <= sqrt(2) + 0.05)
    pairs = skel_tree.query_pairs(r=1.45)
    for i, j in pairs:
        pt1 = skel_points[i]
        pt2 = skel_points[j]
        step_len = np.linalg.norm(pt1 - pt2)
        pixel_adj.add_edge(i, j, weight=step_len)
        
    # 3. Identify pixel indices that correspond to junction regions and temporarily remove them
    # to decompose skeleton into disjoint vessel segments
    junction_node_rows = nodes_df[nodes_df.node_type == "junction"]
    junction_pixel_indices = set()
    
    for _, row in junction_node_rows.iterrows():
        jx, jy = row["x"], row["y"]
        # Find all skeleton pixels within 3.5 px of junction centroid
        nearby = skel_tree.query_ball_point([jx, jy], r=3.5)
        junction_pixel_indices.update(nearby)
        
    # Create segment graph by removing junction pixels
    segment_graph = pixel_adj.copy()
    segment_graph.remove_nodes_from(junction_pixel_indices)
    
    # 4. For each connected component in segment_graph, connect to nearest graph nodes
    for comp in nx.connected_components(segment_graph):
        comp_indices = list(comp)
        if len(comp_indices) == 0:
            continue
            
        pts = skel_points[comp_indices]
        # Order the points along the segment path
        subg = pixel_adj.subgraph(comp_indices)
        
        # Find segment endpoints (nodes in subg with degree <= 1)
        sub_endpoints = [n for n, deg in subg.degree() if deg <= 1]
        if len(sub_endpoints) >= 2:
            start_pix = sub_endpoints[0]
            end_pix = sub_endpoints[-1]
            try:
                ordered_path_indices = nx.shortest_path(subg, start_pix, end_pix)
            except Exception:
                ordered_path_indices = comp_indices
        else:
            ordered_path_indices = comp_indices
            
        path_coords = skel_points[ordered_path_indices]
        
        # Calculate actual path length
        if len(path_coords) > 1:
            diffs = np.diff(path_coords, axis=0)
            actual_len = float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
        else:
            actual_len = 1.0
            
        # Map endpoints of path to nearest nodes in G
        start_pt = path_coords[0]
        end_pt = path_coords[-1]
        
        d_start, nearest_node_1 = node_tree.query(start_pt)
        d_end, nearest_node_2 = node_tree.query(end_pt)
        
        u = node_ids[nearest_node_1]
        v = node_ids[nearest_node_2]
        
        if u != v:
            pos_u = np.array(G.nodes[u]["pos"])
            pos_v = np.array(G.nodes[v]["pos"])
            chord_dist = float(np.linalg.norm(pos_v - pos_u))
            
            # Avoid division by zero
            tortuosity = float(actual_len / chord_dist) if chord_dist > 1e-3 else 1.0
            
            # Avoid adding duplicate or worse edges
            if not G.has_edge(u, v):
                G.add_edge(
                    u, v,
                    path=path_coords.tolist(),
                    length=actual_len,
                    euclidean_distance=chord_dist,
                    tortuosity=tortuosity
                )
                
    # Update actual node degrees in G
    for n in G.nodes():
        G.nodes[n]["degree"] = G.degree[n]
        
    return G


def batch_build_graphs(manifest_df, skeletons_dir: Path, nodes_dir: Path, output_dir: Path = OUTPUTS_DIR) -> None:
    """
    Builds and serializes NetworkX graphs for all images in manifest_df.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        skel_path = skeletons_dir / f"{img_id}_skeleton.png"
        nodes_path = nodes_dir / f"{img_id}_nodes.csv"
        
        skel = np.array(Image.open(skel_path)) > 128
        nodes_df = pd.read_csv(nodes_path)
        
        G = build_vessel_graph(skel, nodes_df)
        
        # Save as JSON (preserves all rich attributes)
        import json
        json_path = output_dir / f"{img_id}_graph.json"
        with open(json_path, "w") as f:
            json.dump(nx.node_link_data(G), f)
            
        # Also save sanitized GraphML
        save_path = output_dir / f"{img_id}_graph.graphml"
        G_export = G.copy()
        for n, d in G_export.nodes(data=True):
            if "pos" in d:
                d["pos_x"] = float(d["pos"][0])
                d["pos_y"] = float(d["pos"][1])
                del d["pos"]
        for u, v, d in G_export.edges(data=True):
            if "path" in d:
                del d["path"]
        nx.write_graphml(G_export, str(save_path))
        
    print(f"[Graph Construction] Constructed vessel graphs for {len(manifest_df)} images -> {output_dir}")
