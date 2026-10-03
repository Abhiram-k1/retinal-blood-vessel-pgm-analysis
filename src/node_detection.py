"""
Node and Branch Candidate Detection from Retinal Vessel Skeletons.
Identifies endpoints (degree 1) and bifurcation/junctions (degree >= 3)
using 8-connected neighborhood cross-numbers and spatial clustering.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "nodes"


def detect_skeleton_nodes(skeleton: np.ndarray, cluster_distance: float = 3.5) -> dict:
    """
    Detects endpoints and junctions from a 1-pixel wide skeleton.
    
    Args:
        skeleton: uint8 or bool array (H, W), 1=skeleton, 0=background
        cluster_distance: distance threshold to merge adjacent junction pixels into single node
        
    Returns:
        dict containing:
            - 'endpoints': list of (x, y) tuples
            - 'junctions': list of (x, y) tuples
            - 'nodes_df': DataFrame of all nodes (node_id, x, y, type, degree)
            - 'neighbor_count_map': uint8 array of 8-neighbor counts
    """
    skel_b = (skeleton > 0).astype(np.uint8)
    
    # 8-neighbor kernel (excluding center)
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)
    
    # Count 8-neighbors
    neighbor_count = cv2.filter2D(skel_b, -1, kernel) * skel_b
    
    # Identify pixel coordinates
    # Endpoints: exactly 1 neighbor
    ep_y, ep_x = np.where((skel_b == 1) & (neighbor_count == 1))
    endpoints = list(zip(ep_x.tolist(), ep_y.tolist()))
    
    # Junction candidates: 3 or more neighbors
    junc_y, junc_x = np.where((skel_b == 1) & (neighbor_count >= 3))
    raw_junction_coords = list(zip(junc_x.tolist(), junc_y.tolist()))
    
    # Cluster adjacent junction pixels to avoid duplicate multi-pixel nodes at single intersection
    clustered_junctions = []
    if len(raw_junction_coords) > 0:
        pts = np.array(raw_junction_coords, dtype=np.float32)
        used = np.zeros(len(pts), dtype=bool)
        
        for i in range(len(pts)):
            if used[i]:
                continue
            # Find all points within cluster_distance
            dists = np.linalg.norm(pts - pts[i], axis=1)
            cluster_idx = np.where(dists <= cluster_distance)[0]
            used[cluster_idx] = True
            centroid = np.mean(pts[cluster_idx], axis=0)
            clustered_junctions.append((int(round(centroid[0])), int(round(centroid[1]))))
            
    # Build unified nodes dataframe
    records = []
    node_id = 0
    for x, y in endpoints:
        records.append({
            "node_id": node_id,
            "x": x,
            "y": y,
            "node_type": "endpoint",
            "degree": 1
        })
        node_id += 1
        
    for x, y in clustered_junctions:
        records.append({
            "node_id": node_id,
            "x": x,
            "y": y,
            "node_type": "junction",
            "degree": 3  # Estimated degree, refined during graph construction
        })
        node_id += 1
        
    nodes_df = pd.DataFrame(records)
    
    return {
        "endpoints": endpoints,
        "junctions": clustered_junctions,
        "nodes_df": nodes_df,
        "neighbor_count_map": neighbor_count
    }


def visualize_detected_nodes(skeleton: np.ndarray, nodes_df: pd.DataFrame) -> np.ndarray:
    """
    Renders visual overlay of detected nodes onto skeleton:
      - Skeleton lines: Light Gray
      - Endpoints: Bright Cyan circles
      - Junctions/Bifurcations: Vibrant Red circles
    """
    h, w = skeleton.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Skeleton in light gray
    vis[skeleton > 0] = [180, 180, 180]
    
    # Draw nodes
    for _, row in nodes_df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        if row["node_type"] == "endpoint":
            cv2.circle(vis, (x, y), radius=4, color=(255, 200, 0), thickness=-1)  # Cyan/Yellow
        else:
            cv2.circle(vis, (x, y), radius=5, color=(0, 50, 255), thickness=-1)   # Red
            
    return vis


def batch_detect_nodes(manifest_df, skeletons_dir: Path, output_dir: Path = OUTPUTS_DIR) -> None:
    """
    Processes all skeletons in skeletons_dir, extracts nodes, and saves data & visualizations.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        skel_path = skeletons_dir / f"{img_id}_skeleton.png"
        skel = np.array(Image.open(skel_path)) > 128
        
        res = detect_skeleton_nodes(skel)
        
        # Save CSV of nodes
        res["nodes_df"].to_csv(output_dir / f"{img_id}_nodes.csv", index=False)
        
        # Save visual map
        vis = visualize_detected_nodes(skel, res["nodes_df"])
        Image.fromarray(vis).save(vis_dir / f"{img_id}_nodes_vis.png")
        
    print(f"[Node Detection] Detected endpoints and junctions for {len(manifest_df)} images -> {output_dir}")
