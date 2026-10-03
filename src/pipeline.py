"""
Master Pipeline Orchestrator for Retinal Vascular Network Analysis.
Executes the full 3-Phase technical workflow:
  Phase 1: Dataset Manifest, Preprocessing, Vessel Segmentation, GT Validation, Skeletonization, Node Detection (100% complete)
  Phase 2: Graph Construction, Geometry, Fractals, Efficiency, Master Feature Table, Bayesian Network, CPTs, Inference, Stats (>=85% complete)
  Phase 3: MRF Energy Modeling, ICM Inference, Baseline vs MRF Benchmarking, Integrated Matrix
"""

import time
from pathlib import Path
import pandas as pd

from src.dataset import build_dataset_manifest
from src.preprocessing import batch_preprocess_dataset
from src.segmentation import batch_segment_dataset
from src.validation import validate_dataset_segmentation
from src.skeletonization import batch_skeletonize_dataset
from src.node_detection import batch_detect_nodes
from src.graph_construction import batch_build_graphs
from src.feature_extraction import extract_all_retinal_features
from src.bayesian_network import train_retinal_bayesian_network
from src.bayesian_inference import evaluate_test_set_inference
from src.statistical_analysis import generate_feature_statistics, generate_correlation_matrix
from src.mrf_inference import run_baseline_vs_mrf_experiment

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_full_pipeline():
    print("================================================================================")
    print("      RETINAL BLOOD-VESSEL PROBABILISTIC & GRAPH ANALYSIS PIPELINE              ")
    print("================================================================================")
    start_total = time.time()
    
    # -------------------------------------------------------------------------
    # PHASE 1: Complete Foundation, Preprocessing, Segmentation, GT Validation, Skeletons
    # -------------------------------------------------------------------------
    print("\n>>> [PHASE 1] Step 1.1: Building Dataset Manifest...")
    manifest_df = build_dataset_manifest()
    
    print("\n>>> [PHASE 1] Step 1.2: Image Preprocessing (Green Channel + CLAHE + Filtering)...")
    preproc_dir = OUTPUTS_DIR / "preprocessed"
    batch_preprocess_dataset(manifest_df, preproc_dir)
    
    print("\n>>> [PHASE 1] Step 1.3: Morphological Vessel Segmentation Engine...")
    masks_dir = OUTPUTS_DIR / "masks"
    batch_segment_dataset(manifest_df, preproc_dir, masks_dir)
    
    print("\n>>> [PHASE 1] Step 1.4: Quantitative Segmentation Validation vs Ground Truth...")
    metrics_df = validate_dataset_segmentation(manifest_df, masks_dir)
    
    print("\n>>> [PHASE 1] Step 1.5: Skeletonization (Medial Axis Centerline Extraction)...")
    skeletons_dir = OUTPUTS_DIR / "skeletons"
    batch_skeletonize_dataset(manifest_df, masks_dir, skeletons_dir)
    
    print("\n>>> [PHASE 1] Step 1.6: Node & Branch Detection (Endpoints & Bifurcations)...")
    nodes_dir = OUTPUTS_DIR / "nodes"
    batch_detect_nodes(manifest_df, skeletons_dir, nodes_dir)
    print(">>> [PHASE 1] 100% COMPLETE! All artifacts generated.")
    
    # -------------------------------------------------------------------------
    # PHASE 2: Graph Theory, Geometry, Fractals, Efficiency & Bayesian Network
    # -------------------------------------------------------------------------
    print("\n>>> [PHASE 2] Step 2.1: Mathematical Graph Construction G=(V, E)...")
    graphs_dir = OUTPUTS_DIR / "graphs"
    batch_build_graphs(manifest_df, skeletons_dir, nodes_dir, graphs_dir)
    
    print("\n>>> [PHASE 2] Step 2.2: Extracting Graph, Geometric, Fractal & Efficiency Features...")
    features_df = extract_all_retinal_features(manifest_df, masks_dir, skeletons_dir, nodes_dir)
    
    print("\n>>> [PHASE 2] Step 2.3: Descriptive Statistics & Correlation Heatmaps...")
    stats_df = generate_feature_statistics(features_df)
    corr_df = generate_correlation_matrix(features_df)
    
    print("\n>>> [PHASE 2] Step 2.4: Training Directed Bayesian Network & Factorization Check...")
    bn_model = train_retinal_bayesian_network(features_df)
    
    print("\n>>> [PHASE 2] Step 2.5: Variable Elimination Probabilistic Inference Engine...")
    bayes_inference_df = evaluate_test_set_inference(bn_model, features_df)
    print(">>> [PHASE 2] >=85% COMPLETE! Mathematical features and Bayesian engine operational.")
    
    # -------------------------------------------------------------------------
    # PHASE 3: MRF Energy Model, ICM Inference & Controlled Baseline Comparison
    # -------------------------------------------------------------------------
    print("\n>>> [PHASE 3] Step 3.1: Undirected MRF Segmentation & Baseline Comparison...")
    mrf_comp_df = run_baseline_vs_mrf_experiment(manifest_df, masks_dir)
    
    print("\n>>> [PHASE 3] Step 3.2: Generating Master Integrated Results Matrix...")
    # Merge mathematical features, Bayesian inference posteriors, and MRF results
    merged = features_df.merge(bayes_inference_df[["image_id", "posterior_prob_normal", "posterior_prob_complex", "inferred_state"]], on="image_id", how="left")
    final_master_df = merged.merge(mrf_comp_df[["image_id", "mrf_initial_energy", "mrf_final_energy", "mrf_accuracy", "mrf_f1", "f1_improvement"]], on="image_id", how="left")
    
    final_master_path = OUTPUTS_DIR / "final_retinal_pgm_results.csv"
    final_master_df.to_csv(final_master_path, index=False)
    print(f"[Master Results] Final combined results table saved -> {final_master_path}")
    
    elapsed = time.time() - start_total
    print("\n================================================================================")
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!                      ")
    print("================================================================================")


if __name__ == "__main__":
    run_full_pipeline()
