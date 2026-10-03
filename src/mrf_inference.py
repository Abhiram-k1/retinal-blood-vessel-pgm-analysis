"""
Markov Random Field (MRF) Inference Module using Iterated Conditional Modes (ICM).
Performs MAP label inference X* = argmin E(X) to obtain spatially regularized vessel segmentations,
and executes the controlled Baseline vs MRF comparison experiment on the held-out test set.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import scipy.signal

from src.markov_network import RetinalMRFEnergyModel
from src.validation import compute_binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def run_icm_inference(
    preproc_float: np.ndarray,
    fov_mask: np.ndarray,
    initial_labels: np.ndarray,
    mrf_model: RetinalMRFEnergyModel,
    max_iters: int = 4
) -> tuple:
    """
    Solves X* = argmin E(X) via Iterated Conditional Modes (ICM).
    
    Args:
        preproc_float: float32 image normalized to [0, 1]
        fov_mask: bool mask indicating valid retinal area
        initial_labels: uint8 array (0 or 1) from classical segmentation baseline
        mrf_model: configured RetinalMRFEnergyModel
        max_iters: number of full lattice update sweeps
        
    Returns:
        (optimal_labels, energy_history)
    """
    d_bg, d_vessel = mrf_model.compute_unary_potentials(preproc_float, fov_mask)
    labels = initial_labels.copy().astype(np.uint8)
    labels[~fov_mask] = 0
    
    energy_history = []
    initial_energy = mrf_model.compute_total_energy(labels, d_bg, d_vessel, fov_mask)
    energy_history.append(initial_energy)
    
    # 4-connected spatial convolution kernel
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # Precompute neighbor counts within FOV
    fov_float = fov_mask.astype(np.float32)
    total_valid_neighbors = scipy.signal.convolve2d(fov_float, kernel, mode="same")
    
    for it in range(max_iters):
        # Count current vessel neighbors for all pixels
        vessel_neighbors = scipy.signal.convolve2d(labels.astype(np.float32), kernel, mode="same")
        bg_neighbors = total_valid_neighbors - vessel_neighbors
        
        # Local energy for choosing label 0 (Background) vs label 1 (Vessel)
        # Cost if label = 0: unary d_bg + lambda * beta * (number of vessel neighbors)
        e_label_0 = d_bg + mrf_model.lambda_weight * mrf_model.beta_smooth * vessel_neighbors
        
        # Cost if label = 1: unary d_vessel + lambda * beta * (number of background neighbors)
        e_label_1 = d_vessel + mrf_model.lambda_weight * mrf_model.beta_smooth * bg_neighbors
        
        # Update rule: pick label with lower energy inside FOV
        new_labels = np.where((e_label_1 < e_label_0) & fov_mask, 1, 0).astype(np.uint8)
        
        # Check convergence
        changed_pixels = np.sum(new_labels != labels)
        labels = new_labels
        
        current_energy = mrf_model.compute_total_energy(labels, d_bg, d_vessel, fov_mask)
        energy_history.append(current_energy)
        
        if changed_pixels < 50:
            break
            
    return labels, energy_history


def run_baseline_vs_mrf_experiment(manifest_df, masks_dir: Path, output_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Executes controlled Experiment comparing Classical Morphological Baseline vs MRF Segmentation
    on the held-out test partition.
    """
    mrf_model = RetinalMRFEnergyModel(lambda_weight=1.5, beta_smooth=1.0)
    mrf_masks_dir = output_dir / "mrf_masks"
    mrf_masks_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    test_df = manifest_df[manifest_df.split == "test"].copy()
    
    for idx, row in test_df.iterrows():
        img_id = row["image_id"]
        preproc_path = output_dir / "preprocessed" / f"{img_id}_preprocessed.png"
        baseline_mask_path = masks_dir / f"{img_id}_vessel_mask.png"
        
        preproc_img = np.array(Image.open(preproc_path)).astype(np.float32) / 255.0
        baseline_mask = (np.array(Image.open(baseline_mask_path)) > 128).astype(np.uint8)
        
        gt1_mask = np.array(Image.open(row["manual1_path"])) > 128
        if gt1_mask.ndim == 3:
            gt1_mask = gt1_mask[:, :, 0]
            
        fov_mask = np.array(Image.open(row["mask_path"])) > 128
        if fov_mask.ndim == 3:
            fov_mask = fov_mask[:, :, 0]
            
        # Run ICM inference
        mrf_mask, energy_hist = run_icm_inference(preproc_img, fov_mask, baseline_mask, mrf_model)
        
        # Save MRF mask
        Image.fromarray(mrf_mask * 255).save(mrf_masks_dir / f"{img_id}_mrf_mask.png")
        
        # Compute baseline vs MRF metrics
        base_res = compute_binary_metrics(baseline_mask, gt1_mask, fov_mask)
        mrf_res = compute_binary_metrics(mrf_mask, gt1_mask, fov_mask)
        
        records.append({
            "image_id": img_id,
            "baseline_accuracy": base_res["accuracy"],
            "baseline_precision": base_res["precision"],
            "baseline_recall": base_res["recall_sensitivity"],
            "baseline_specificity": base_res["specificity"],
            "baseline_f1": base_res["f1_score"],
            "mrf_initial_energy": energy_hist[0],
            "mrf_final_energy": energy_hist[-1],
            "mrf_accuracy": mrf_res["accuracy"],
            "mrf_precision": mrf_res["precision"],
            "mrf_recall": mrf_res["recall_sensitivity"],
            "mrf_specificity": mrf_res["specificity"],
            "mrf_f1": mrf_res["f1_score"],
            "f1_improvement": mrf_res["f1_score"] - base_res["f1_score"]
        })
        
    comp_df = pd.DataFrame(records)
    csv_path = output_dir / "baseline_vs_mrf_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    
    print(f"[MRF Experiment] Baseline vs MRF comparison completed ({len(comp_df)} test images) -> {csv_path}")
    print(f"  Test Mean Baseline F1: {comp_df['baseline_f1'].mean():.4f}")
    print(f"  Test Mean MRF F1:      {comp_df['mrf_f1'].mean():.4f}")
    print(f"  Mean F1 Improvement:   {comp_df['f1_improvement'].mean():+.4f}")
    return comp_df
