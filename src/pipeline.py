"""
Master Pipeline Orchestrator for Retinal Vascular Network Analysis.

Executes the Unified End-to-End Technical Workflow:
  1. Dataset Ingestion & Patient-Level Split (CHASE_DB1)
  2. Image Preprocessing (Green Channel, CLAHE, Noise Filtering)
  3. Multiresolution Analysis (MRA via Multi-Scale Gabor Filterbank)
  4. Deep Learning Vessel Segmentation (Neural Receptive Field Network)
  5. Morphological Baseline Segmentation & Ground Truth Validation (Dice, IoU, F1)
  6. Centerline Skeletonization & Bifurcation/Endpoint Node Detection
  7. Mathematical Topological Graph Construction G=(V,E) (with Gap Bridging)
  8. Graph-Theoretic, Geometric, Fractal, and Efficiency Feature Extraction
  9. Descriptive Statistics & Correlation Heatmap Generation
  10. Directed Bayesian Network Training (Non-Circular Fractal Target State)
  11. Exact Variable Elimination Bayesian Posterior Inference Engine
  12. Undirected Markov Random Field (MRF) Spatial Regularization & ICM Inference
  13. Comprehensive 5-Condition Ablation Study
  14. Master Integrated Results Matrix Generation & Figure Visualization
"""

import time
from pathlib import Path
import pandas as pd

from src.dataset import build_dataset_manifest
from src.preprocessing import batch_preprocess_dataset
from src.multiscale_features import batch_compute_gabor_features
from src.deep_learning import train_and_evaluate_deep_model
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
from src.ablation_study import run_comprehensive_ablation_study
from src.visualization import main as render_figures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_full_pipeline():
    print("================================================================================")
    print("      RETINAL BLOOD-VESSEL PROBABILISTIC & GRAPH ANALYSIS PIPELINE              ")
    print("================================================================================")
    start_total = time.time()
    
    # -------------------------------------------------------------------------
    # STAGE 1: Dataset & Preprocessing
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 1] Step 1.1: Building Dataset Manifest...")
    manifest_df = build_dataset_manifest()
    
    print("\n>>> [STAGE 1] Step 1.2: Image Preprocessing (Green Channel + CLAHE + Filtering)...")
    preproc_dir = OUTPUTS_DIR / "preprocessed"
    batch_preprocess_dataset(manifest_df, preproc_dir)

    # -------------------------------------------------------------------------
    # STAGE 2: Multiresolution Analysis (MRA) & Deep Learning
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 2] Step 2.1: Multiresolution Analysis (Multi-Scale Gabor Filterbank)...")
    batch_compute_gabor_features(manifest_df, preproc_dir, OUTPUTS_DIR)

    print("\n>>> [STAGE 2] Step 2.2: Deep Learning Neural Network Training & Inference...")
    dl_model, dl_metrics_df = train_and_evaluate_deep_model(manifest_df, preproc_dir, OUTPUTS_DIR)
    
    # -------------------------------------------------------------------------
    # STAGE 3: Baseline Segmentation, Validation, Skeletons, & Nodes
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 3] Step 3.1: Morphological Vessel Segmentation Engine...")
    masks_dir = OUTPUTS_DIR / "masks"
    batch_segment_dataset(manifest_df, preproc_dir, masks_dir)
    
    print("\n>>> [STAGE 3] Step 3.2: Quantitative Segmentation Validation vs Ground Truth...")
    metrics_df = validate_dataset_segmentation(manifest_df, masks_dir)
    
    print("\n>>> [STAGE 3] Step 3.3: Skeletonization (Medial Axis Centerline Extraction)...")
    skeletons_dir = OUTPUTS_DIR / "skeletons"
    batch_skeletonize_dataset(manifest_df, masks_dir, skeletons_dir)
    
    print("\n>>> [STAGE 3] Step 3.4: Node & Branch Detection (Endpoints & Bifurcations)...")
    nodes_dir = OUTPUTS_DIR / "nodes"
    batch_detect_nodes(manifest_df, skeletons_dir, nodes_dir)
    
    # -------------------------------------------------------------------------
    # STAGE 4: Mathematical Graph Theory, Geometry, Fractals, & Efficiency
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 4] Step 4.1: Mathematical Graph Construction G=(V, E)...")
    graphs_dir = OUTPUTS_DIR / "graphs"
    batch_build_graphs(manifest_df, skeletons_dir, nodes_dir, graphs_dir)
    
    print("\n>>> [STAGE 4] Step 4.2: Extracting Graph, Geometric, Fractal & Efficiency Features...")
    features_df = extract_all_retinal_features(manifest_df, masks_dir, skeletons_dir, nodes_dir)
    
    print("\n>>> [STAGE 4] Step 4.3: Descriptive Statistics & Correlation Heatmaps...")
    stats_df = generate_feature_statistics(features_df)
    corr_df = generate_correlation_matrix(features_df)
    
    # -------------------------------------------------------------------------
    # STAGE 5: Directed Bayesian Network Modeling & Posterior Inference
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 5] Step 5.1: Training Directed Bayesian Network & Factorization Check...")
    bn_model = train_retinal_bayesian_network(features_df)
    
    print("\n>>> [STAGE 5] Step 5.2: Variable Elimination Probabilistic Inference Engine...")
    bayes_inference_df = evaluate_test_set_inference(bn_model, features_df)
    
    # -------------------------------------------------------------------------
    # STAGE 6: Undirected MRF Spatial Regularization & Controlled Benchmark
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 6] Step 6.1: Undirected MRF Spatial Regularization & Benchmark...")
    mrf_comp_df = run_baseline_vs_mrf_experiment(manifest_df, masks_dir)

    # -------------------------------------------------------------------------
    # STAGE 7: Comprehensive Ablation Study (5 Conditions)
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 7] Step 7.1: Comprehensive Ablation Study...")
    ablation_df = run_comprehensive_ablation_study(manifest_df, dl_model)
    
    # -------------------------------------------------------------------------
    # STAGE 8: Master Integrated Results Matrix & Visualizations
    # -------------------------------------------------------------------------
    print("\n>>> [STAGE 8] Step 8.1: Generating Master Integrated Results Matrix...")
    merged = features_df.merge(
        bayes_inference_df[["image_id", "posterior_prob_normal", "posterior_prob_complex", "inferred_state"]],
        on="image_id",
        how="left"
    )
    merged = merged.merge(
        mrf_comp_df[["image_id", "mrf_initial_energy", "mrf_final_energy", "mrf_accuracy", "mrf_f1", "f1_improvement"]],
        on="image_id",
        how="left"
    )
    final_master_df = merged.merge(
        dl_metrics_df[["image_id", "dl_accuracy", "dl_precision", "dl_recall", "dl_f1", "dl_dice", "dl_iou"]],
        on="image_id",
        how="left"
    )
    
    final_master_path = OUTPUTS_DIR / "final_retinal_pgm_results.csv"
    final_master_df.to_csv(final_master_path, index=False)
    print(f"[Master Results] Final combined results table saved -> {final_master_path}")
    
    print("\n>>> [STAGE 8] Step 8.2: Rendering High-Resolution Result Visualizations...")
    render_figures()
    
    elapsed = time.time() - start_total
    print("\n================================================================================")
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!                      ")
    print("================================================================================")


if __name__ == "__main__":
    run_full_pipeline()
