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
                    "This interactive notebook demonstrates the complete end-to-end pipeline:\n",
                    "1. **Dataset Ingestion & Manifest Verification** (CHASE_DB1 28 images: 20 Train, 8 Test)\n",
                    "2. **Image Preprocessing** (Green Channel, CLAHE, Bilateral Filtering, FOV Masking)\n",
                    "3. **Morphological Vessel Segmentation & Ground Truth Validation** (Pixel-wise Confusion Matrix, F1, Accuracy, Sensitivity, Specificity)\n",
                    "4. **Skeletonization & Topological Node Detection** (Medial Axis Thinning, Endpoints, Bifurcations)\n",
                    "5. **Mathematical Graph Construction $G=(V, E)$** (NetworkX)\n",
                    "6. **Vascular Geometric & Morphological Analysis** (True Curve Length, Chord Distance, Tortuosity, Branch Angles)\n",
                    "7. **Fractal Dimension Analysis** (Multi-Scale Box-Counting, Log-Log Slope)\n",
                    "8. **Global Network Efficiency** (All-Pairs Shortest Paths with Disconnected Handling)\n",
                    "9. **Directed Bayesian Network** (Empirical Tertile Discretization, DAG Architecture, CPT Learning, Factorization Proof, Exact Variable Elimination)\n",
                    "10. **Undirected Markov Random Field (MRF)** (Spatial Energy Model, Iterated Conditional Modes (ICM), Controlled Baseline vs MRF Comparison)\n",
                    "11. **Statistical Summaries & Correlation Matrices**"
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
                    "# Ensure src is on sys.path\n",
                    "sys.path.append(str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))\n",
                    "print('Environment and modules loaded successfully!')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Dataset Manifest & Patient-Level Split Verification"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "manifest_path = Path('../outputs/dataset_manifest.csv') if Path('../outputs').exists() else Path('outputs/dataset_manifest.csv')\n",
                    "manifest_df = pd.read_csv(manifest_path)\n",
                    "print(f'Total records in CHASE_DB1: {len(manifest_df)}')\n",
                    "print(f'Training count: {len(manifest_df[manifest_df.split==\"train\"])}')\n",
                    "print(f'Testing count:  {len(manifest_df[manifest_df.split==\"test\"])}')\n",
                    "manifest_df[['image_id', 'patient_id', 'eye', 'split']].head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Image Preprocessing & Contrast Enhancement\n",
                    "Displaying Original RGB Fundus, Isolated Green Channel ($I_G$), and CLAHE Contrast-Enhanced Normalized Image."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sample = manifest_df.iloc[0]\n",
                    "orig_img = Image.open(sample['image_path'])\n",
                    "orig_np = np.array(orig_img)\n",
                    "green_np = orig_np[:, :, 1]\n",
                    "\n",
                    "preproc_path = Path(f'../outputs/preprocessed/{sample[\"image_id\"]}_preprocessed.png') if Path('../outputs').exists() else Path(f'outputs/preprocessed/{sample[\"image_id\"]}_preprocessed.png')\n",
                    "preproc_np = np.array(Image.open(preproc_path))\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                    "axes[0].imshow(orig_np)\n",
                    "axes[0].set_title(f'Original RGB: {sample[\"image_id\"]}')\n",
                    "axes[0].axis('off')\n",
                    "\n",
                    "axes[1].imshow(green_np, cmap='gray')\n",
                    "axes[1].set_title('Green Channel (Hemoglobin Absorption)')\n",
                    "axes[1].axis('off')\n",
                    "\n",
                    "axes[2].imshow(preproc_np, cmap='gray')\n",
                    "axes[2].set_title('CLAHE + Bilateral Filtering')\n",
                    "axes[2].axis('off')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Vessel Segmentation & Quantitative Validation Against Ground Truth"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "metrics_path = Path('../outputs/segmentation_metrics.csv') if Path('../outputs').exists() else Path('outputs/segmentation_metrics.csv')\n",
                    "metrics_df = pd.read_csv(metrics_path)\n",
                    "print('Overall Segmentation Validation Averages (Observer 1 Ground Truth):')\n",
                    "print(metrics_df[['obs1_accuracy', 'obs1_precision', 'obs1_recall', 'obs1_specificity', 'obs1_f1']].mean().to_frame(name='Mean Metric'))\n",
                    "metrics_df.head(6)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Skeletonization & Topological Node Detection\n",
                    "1-Pixel Medial Axis Centerlines, Endpoints (Yellow/Cyan), and Bifurcations (Red)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "skel_overlay_path = Path(f'../outputs/skeletons/overlays/{sample[\"image_id\"]}_skel_overlay.png') if Path('../outputs').exists() else Path(f'outputs/skeletons/overlays/{sample[\"image_id\"]}_skel_overlay.png')\n",
                    "node_vis_path = Path(f'../outputs/nodes/visualizations/{sample[\"image_id\"]}_nodes_vis.png') if Path('../outputs').exists() else Path(f'outputs/nodes/visualizations/{sample[\"image_id\"]}_nodes_vis.png')\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(12, 6))\n",
                    "axes[0].imshow(Image.open(skel_overlay_path))\n",
                    "axes[0].set_title('Binary Mask (Blue) vs Centerline Skeleton (Red)')\n",
                    "axes[0].axis('off')\n",
                    "\n",
                    "axes[1].imshow(Image.open(node_vis_path))\n",
                    "axes[1].set_title('Detected Nodes: Endpoints (Cyan) & Junctions (Red)')\n",
                    "axes[1].axis('off')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Master Mathematical Feature Table\n",
                    "Graph Topology, Tortuosity, Fractal Dimension, and Global Efficiency."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "feat_path = Path('../outputs/retinal_network_features.csv') if Path('../outputs').exists() else Path('outputs/retinal_network_features.csv')\n",
                    "feat_df = pd.read_csv(feat_path)\n",
                    "feat_df[['image_id', 'nodes', 'edges', 'branches', 'density', 'mean_tortuosity', 'fractal_dimension', 'global_efficiency']].head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Bayesian Network: CPT Learning & Posterior Inference\n",
                    "Directed Acyclic Graph (DAG) with Exact Variable Elimination."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "bn_res_path = Path('../outputs/bayesian_inference_results.csv') if Path('../outputs').exists() else Path('outputs/bayesian_inference_results.csv')\n",
                    "bn_res_df = pd.read_csv(bn_res_path)\n",
                    "print('Bayesian Inferred State Posteriors across Retinal Images:')\n",
                    "bn_res_df[['image_id', 'split', 'observed_tortuosity', 'observed_branching', 'posterior_prob_normal', 'posterior_prob_complex', 'inferred_state']].head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Markov Random Field (MRF): Energy Minimization & Baseline Benchmarking"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "mrf_comp_path = Path('../outputs/baseline_vs_mrf_comparison.csv') if Path('../outputs').exists() else Path('outputs/baseline_vs_mrf_comparison.csv')\n",
                    "mrf_comp_df = pd.read_csv(mrf_comp_path)\n",
                    "print('Held-Out Test Set: Morphological Baseline vs Undirected MRF:')\n",
                    "mrf_comp_df[['image_id', 'baseline_f1', 'mrf_f1', 'f1_improvement', 'mrf_initial_energy', 'mrf_final_energy']]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 8. Feature Correlation Matrix Heatmap"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "corr_img_path = Path('../outputs/plots/correlation_matrix.png') if Path('../outputs').exists() else Path('outputs/plots/correlation_matrix.png')\n",
                    "if corr_img_path.exists():\n",
                    "    plt.figure(figsize=(10, 8))\n",
                    "    plt.imshow(Image.open(corr_img_path))\n",
                    "    plt.axis('off')\n",
                    "    plt.show()"
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
