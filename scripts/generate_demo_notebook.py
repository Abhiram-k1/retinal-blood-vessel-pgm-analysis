"""
Generator script to build a complete, self-contained demonstration Jupyter Notebook.
Outputs to notebooks/retinal_pgm_pipeline_demo.ipynb.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_demo_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Probabilistic and Mathematical Analysis of Retinal Blood-Vessel Networks\n",
                    "### Course Project: Probabilistic Graphical Models (PGM) | Alignment: CO1–CO4\n",
                    "\n",
                    "This interactive notebook demonstrates the complete end-to-end research pipeline:\n",
                    "1. **Dataset Ingestion & Patient-Level Split** (CHASE_DB1 28 images: 20 Train, 8 Test)\n",
                    "2. **Image Preprocessing** (Green Channel, CLAHE, Bilateral Filtering, FOV Masking)\n",
                    "3. **Multiresolution Analysis (MRA)** (Multi-Scale Directional Gabor Filterbank)\n",
                    "4. **Supervised Deep Learning Model** (Neural Receptive Fields with 10D Multi-Scale Features)\n",
                    "5. **Morphological Vessel Segmentation & Ground Truth Validation** (Confusion Matrix, F1, Dice, IoU)\n",
                    "6. **Skeletonization & Topological Node Detection** (Medial Axis Thinning, Endpoints, Bifurcations)\n",
                    "7. **Mathematical Graph Construction $G=(V, E)$** (Gap-Bridging, Fragmentation Reduction)\n",
                    "8. **Vascular Geometry & Murray's Law** (True Curve Length, Chord Distance, Tortuosity, Branch Angles)\n",
                    "9. **Fractal Dimension Analysis** (Multi-Scale Box-Counting, Log-Log Slope)\n",
                    "10. **Global Network Efficiency** (All-Pairs Shortest Paths with Disconnected Handling)\n",
                    "11. **Directed Bayesian Network** (Non-Circular Discretization, DAG, CPTs, Exact Variable Elimination)\n",
                    "12. **Undirected Markov Random Field (MRF)** (Spatial Energy Model, Data-Driven Likelihoods, ICM)\n",
                    "13. **Comprehensive 5-Condition Ablation Study** (Baseline vs MRA vs MRF vs DL vs Full System)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup and Imports\n",
                    "import os\n",
                    "import sys\n",
                    "from pathlib import Path\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from PIL import Image\n",
                    "\n",
                    "# Add project root to path\n",
                    "root_dir = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(root_dir))\n",
                    "print(f'Working directory: {root_dir}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Manifest & Leakage-Free Patient Split"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "manifest_path = root_dir / 'outputs' / 'dataset_manifest.csv'\n",
                    "manifest_df = pd.read_csv(manifest_path)\n",
                    "print(f'Total Images: {len(manifest_df)} (Train: {len(manifest_df[manifest_df.split==\"train\"])} | Test: {len(manifest_df[manifest_df.split==\"test\"])})')\n",
                    "manifest_df[['image_id', 'patient_id', 'eye', 'split']].head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Quantitative Segmentation Performance vs Ground Truth"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "seg_metrics_path = root_dir / 'outputs' / 'segmentation_metrics.csv'\n",
                    "seg_df = pd.read_csv(seg_metrics_path)\n",
                    "print('Test Set Segmentation Validation against Observer 1:')\n",
                    "test_seg = seg_df[seg_df.split == 'test']\n",
                    "test_seg[['image_id', 'obs1_accuracy', 'obs1_precision', 'obs1_recall', 'obs1_specificity', 'obs1_f1']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Deep Learning Neural Receptive-Field Model Results"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dl_metrics_path = root_dir / 'outputs' / 'deep_learning_metrics.csv'\n",
                    "if dl_metrics_path.exists():\n",
                    "    dl_df = pd.read_csv(dl_metrics_path)\n",
                    "    print('Held-Out Test Set Deep Learning Evaluation:')\n",
                    "    display(dl_df[['image_id', 'dl_accuracy', 'dl_precision', 'dl_recall', 'dl_f1', 'dl_dice', 'dl_iou']])\n",
                    "    print(f'Mean Test Recall: {dl_df[\"dl_recall\"].mean():.4f} | Mean Test Dice/F1: {dl_df[\"dl_dice\"].mean():.4f}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Comprehensive 5-Condition Ablation Study"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ablation_path = root_dir / 'outputs' / 'ablation_study_summary.csv'\n",
                    "if ablation_path.exists():\n",
                    "    abl_df = pd.read_csv(ablation_path)\n",
                    "    display(abl_df)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Topological Graph Metrics & Vascular Geometry (Murray's Law)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "features_path = root_dir / 'outputs' / 'retinal_network_features.csv'\n",
                    "features_df = pd.read_csv(features_path)\n",
                    "print('Extracted Vascular Graph Features (first 8 images):')\n",
                    "features_df[['image_id', 'nodes', 'edges', 'connected_components', 'density', 'mean_angle', 'mean_tortuosity', 'fractal_dimension']].head(8)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Directed Bayesian Network: Exact Variable Elimination Inference"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "bn_res_path = root_dir / 'outputs' / 'bayesian_inference_results.csv'\n",
                    "bn_res_df = pd.read_csv(bn_res_path)\n",
                    "print('Bayesian Inferred State Posteriors across Retinal Images:')\n",
                    "bn_res_df[['image_id', 'split', 'observed_tortuosity', 'observed_branching', 'posterior_prob_normal', 'posterior_prob_complex', 'inferred_state']].head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Markov Random Field (MRF): Energy Minimization & Benchmarking"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "mrf_comp_path = root_dir / 'outputs' / 'baseline_vs_mrf_comparison.csv'\n",
                    "mrf_comp_df = pd.read_csv(mrf_comp_path)\n",
                    "print('Held-Out Test Set: Morphological Baseline vs Undirected MRF:')\n",
                    "mrf_comp_df[['image_id', 'baseline_f1', 'mrf_f1', 'f1_improvement', 'mrf_initial_energy', 'mrf_final_energy']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Master Visual Gallery"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig_files = [\n",
                    "    root_dir / 'outputs' / 'figures' / '00_results_dashboard.png',\n",
                    "    root_dir / 'outputs' / 'figures' / '15_ablation_study_comparison.png',\n",
                    "    root_dir / 'outputs' / 'figures' / '01_pipeline_stages.png'\n",
                    "]\n",
                    "for fig_p in fig_files:\n",
                    "    if fig_p.exists():\n",
                    "        plt.figure(figsize=(12, 6))\n",
                    "        plt.imshow(Image.open(fig_p))\n",
                    "        plt.axis('off')\n",
                    "        plt.title(fig_p.name, fontsize=12, fontweight='bold')\n",
                    "        plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    nb_path = NOTEBOOKS_DIR / "retinal_pgm_pipeline_demo.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[Notebook] Demo Jupyter Notebook generated -> {nb_path}")


if __name__ == "__main__":
    create_demo_notebook()
