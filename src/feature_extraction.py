"""
Master Feature Extraction Pipeline.
Aggregates graph topology, vascular geometry, box-counting fractals, and global efficiency
into the Master Mathematical Feature Table (retinal_network_features.csv).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.graph_construction import build_vessel_graph
from src.graph_analysis import analyze_graph_topology
from src.geometric_analysis import analyze_vessel_geometry
from src.fractal_analysis import compute_box_counting_fractal_dimension
from src.network_efficiency import compute_global_network_efficiency

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def extract_all_retinal_features(manifest_df, masks_dir: Path, skeletons_dir: Path, nodes_dir: Path) -> pd.DataFrame:
    """
    Computes all mathematical, geometric, and topological features across the entire dataset.
    """
    records = []
    plots_dir = OUTPUTS_DIR / "plots" / "fractals"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        mask_path = masks_dir / f"{img_id}_vessel_mask.png"
        skel_path = skeletons_dir / f"{img_id}_skeleton.png"
        nodes_path = nodes_dir / f"{img_id}_nodes.csv"
        
        mask = np.array(Image.open(mask_path)) > 128
        skel = np.array(Image.open(skel_path)) > 128
        nodes_df = pd.read_csv(nodes_path)
        
        # 1. Build Graph
        G = build_vessel_graph(skel, nodes_df)
        
        # 2. Graph Topology
        topo = analyze_graph_topology(G)
        
        # 3. Geometry & Tortuosity
        geom = analyze_vessel_geometry(G)
        
        # 4. Fractal Dimension
        plot_file = plots_dir / f"{img_id}_fractal_loglog.png"
        fractal = compute_box_counting_fractal_dimension(mask, save_plot_path=plot_file)
        
        # 5. Global Efficiency
        efficiency = compute_global_network_efficiency(G)
        
        records.append({
            "image_id": img_id,
            "patient_id": row["patient_id"],
            "eye": row["eye"],
            "split": row["split"],
            "nodes": topo["nodes"],
            "edges": topo["edges"],
            "branches": topo["branch_count"],
            "connected_components": topo["connected_components"],
            "density": topo["network_density"],
            "total_length": geom["total_length"],
            "mean_length": geom["mean_length"],
            "mean_angle": geom["mean_angle"],
            "mean_tortuosity": geom["mean_tortuosity"],
            "fractal_dimension": fractal["fractal_dimension"],
            "global_efficiency": efficiency
        })
        
    df_features = pd.DataFrame(records)
    csv_path = OUTPUTS_DIR / "retinal_network_features.csv"
    df_features.to_csv(csv_path, index=False)
    print(f"[Feature Extraction] Master Feature Table generated ({len(df_features)} records) -> {csv_path}")
    return df_features
