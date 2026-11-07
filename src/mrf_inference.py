"""
MRF Inference using Iterated Conditional Modes (ICM).
BUG-001 FIX: fit_likelihoods() is called on training images.
BUG-002 FIX: ICM uses a vectorised-but-damped update (checkerboard pattern) 
             which is an accepted practical approximation that avoids label oscillation 
             while remaining computationally tractable on 999x960 images.
Lambda sweep on training set selects optimal regularisation weight.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.markov_network import RetinalMRFEnergyModel
from src.validation import compute_binary_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


def run_icm_inference(
    preproc_float:  np.ndarray,
    fov_mask:       np.ndarray,
    initial_labels: np.ndarray,
    mrf_model:      RetinalMRFEnergyModel,
    max_iters:      int = 8,
) -> tuple:
    """
    Solves X* = argmin E(X) via checkerboard Iterated Conditional Modes (ICM).

    Checkerboard (red/black) decomposition allows vectorised updates while
    respecting the conditional independence structure of the 4-connected MRF:
    at each half-step every updated pixel has only fixed (not simultaneously
    updated) neighbours, so each individual update is locally optimal.

    Args:
        preproc_float  : float32 image in [0,1], shape (H,W)
        fov_mask       : bool mask (H,W), True = inside retinal disc
        initial_labels : uint8 array (H,W), 0/1 from baseline segmentation
        mrf_model      : configured RetinalMRFEnergyModel with fitted likelihoods
        max_iters      : number of full checkerboard sweeps

    Returns:
        (optimal_labels, energy_history)
    """
    d_bg, d_vessel = mrf_model.compute_unary_potentials(preproc_float, fov_mask)
    labels = initial_labels.copy().astype(np.int32)
    labels[~fov_mask.astype(bool)] = 0

    H, W = labels.shape
    lam  = mrf_model.lambda_weight * mrf_model.beta_smooth

    energy_history = [mrf_model.compute_total_energy(labels, d_bg, d_vessel, fov_mask)]

    # Build checkerboard index masks once
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    red_mask   = ((rows + cols) % 2 == 0) & fov_mask.astype(bool)
    black_mask = ((rows + cols) % 2 == 1) & fov_mask.astype(bool)

    for it in range(max_iters):
        changed = 0
        for mask in (red_mask, black_mask):
            # Count vessel-labelled neighbours for pixels in current colour
            # 4-connected neighbours: up, down, left, right
            nbr_sum = np.zeros((H, W), dtype=np.float32)
            nbr_sum[1:,  :] += labels[:-1, :]  # up
            nbr_sum[:-1, :] += labels[1:,  :]  # down
            nbr_sum[:,  1:] += labels[:, :-1]  # left
            nbr_sum[:, :-1] += labels[:, 1: ]  # right

            # Number of valid FOV neighbours
            nbr_count = np.zeros((H, W), dtype=np.float32)
            fov_f = fov_mask.astype(np.float32)
            nbr_count[1:,  :] += fov_f[:-1, :]
            nbr_count[:-1, :] += fov_f[1:,  :]
            nbr_count[:,  1:] += fov_f[:, :-1]
            nbr_count[:, :-1] += fov_f[:, 1: ]

            bg_nbrs     = nbr_count - nbr_sum   # neighbours labelled 0

            # Local energy: cost of assigning label 0 vs label 1
            e0 = d_bg     + lam * nbr_sum     # bg cost: penalise vessel neighbours
            e1 = d_vessel + lam * bg_nbrs      # vessel cost: penalise bg neighbours

            new_label = np.where(e1 < e0, 1, 0).astype(np.int32)

            # Apply update only to pixels in current colour mask
            prev = labels[mask]
            labels[mask] = new_label[mask]
            changed += int(np.sum(labels[mask] != prev))

        energy_history.append(
            mrf_model.compute_total_energy(labels, d_bg, d_vessel, fov_mask)
        )
        if changed < 100:
            break

    return labels.astype(np.uint8), energy_history


def _sweep_lambda_on_train(manifest_df, preproc_dir, masks_dir, fov_cache):
    """
    Evaluates MRF F1 on training images for lambda in [0.3, 0.6, 1.0, 1.5, 2.0].
    Returns best lambda and fitted MRF model.
    """
    train_df = manifest_df[manifest_df["split"] == "train"]
    lambdas  = [0.3, 0.6, 1.0, 1.5, 2.0]
    best_lambda = 1.0
    best_f1     = -1.0

    # Pre-load training images / masks / fov for fitting
    train_preprocs, train_bin_masks, train_fovs = [], [], []
    for _, row in train_df.iterrows():
        img_id = row["image_id"]
        pp = np.array(Image.open(preproc_dir / f"{img_id}_preprocessed.png")).astype(np.float32) / 255.0
        bm = (np.array(Image.open(masks_dir / f"{img_id}_vessel_mask.png")) > 128).astype(np.uint8)
        fv = fov_cache[img_id]
        train_preprocs.append(pp)
        train_bin_masks.append(bm)
        train_fovs.append(fv)

    for lam in lambdas:
        mrf = RetinalMRFEnergyModel(lambda_weight=lam, beta_smooth=0.8)
        mrf.fit_likelihoods(train_preprocs, train_bin_masks, train_fovs)

        f1_scores = []
        for pp, bm, fv, (_, row) in zip(train_preprocs, train_bin_masks, train_fovs, train_df.iterrows()):
            img_id = row["image_id"]
            gt1    = np.array(Image.open(row["manual1_path"])) > 128
            if gt1.ndim == 3:
                gt1 = gt1[:, :, 0]
            mrf_mask, _ = run_icm_inference(pp, fv, bm, mrf, max_iters=4)
            m = compute_binary_metrics(mrf_mask, gt1, fv)
            f1_scores.append(m["f1_score"])

        mean_f1 = float(np.mean(f1_scores))
        print(f"[MRF Lambda Sweep] lambda={lam:.1f} -> train F1={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1     = mean_f1
            best_lambda = lam

    print(f"[MRF Lambda Sweep] Best lambda={best_lambda}, F1={best_f1:.4f}")
    # Refit final model with best lambda
    best_mrf = RetinalMRFEnergyModel(lambda_weight=best_lambda, beta_smooth=0.8)
    best_mrf.fit_likelihoods(train_preprocs, train_bin_masks, train_fovs)
    return best_mrf


def run_baseline_vs_mrf_experiment(
    manifest_df,
    masks_dir:  Path,
    output_dir: Path = OUTPUTS_DIR
) -> pd.DataFrame:
    """
    Controlled experiment: Classical Morphological Baseline vs MRF on held-out test set.

    Steps:
    1. Load all training images to fit MRF Gaussian likelihoods (BUG-001 fix).
    2. Sweep lambda on training set to find optimal regularisation (BUG-002 fix).
    3. Run corrected checkerboard ICM on the 8 test images.
    4. Report per-image and mean metrics.
    """
    preproc_dir  = output_dir / "preprocessed"
    mrf_masks_dir = output_dir / "mrf_masks"
    mrf_masks_dir.mkdir(parents=True, exist_ok=True)

    # Cache FOV masks for all images
    fov_cache = {}
    for _, row in manifest_df.iterrows():
        fv = np.array(Image.open(row["mask_path"])) > 128
        if fv.ndim == 3:
            fv = fv[:, :, 0]
        fov_cache[row["image_id"]] = fv

    # Step 1 & 2: fit likelihoods and sweep lambda on training set
    print("[MRF] Fitting likelihoods and sweeping lambda on training set...")
    best_mrf = _sweep_lambda_on_train(manifest_df, preproc_dir, masks_dir, fov_cache)

    # Step 3: evaluate on test set
    records  = []
    test_df  = manifest_df[manifest_df["split"] == "test"]

    for _, row in test_df.iterrows():
        img_id = row["image_id"]
        pp     = np.array(Image.open(preproc_dir / f"{img_id}_preprocessed.png")).astype(np.float32) / 255.0
        bm     = (np.array(Image.open(masks_dir  / f"{img_id}_vessel_mask.png")) > 128).astype(np.uint8)
        fv     = fov_cache[img_id]
        gt1    = np.array(Image.open(row["manual1_path"])) > 128
        if gt1.ndim == 3:
            gt1 = gt1[:, :, 0]

        mrf_mask, energy_hist = run_icm_inference(pp, fv, bm, best_mrf, max_iters=8)
        Image.fromarray((mrf_mask * 255).astype(np.uint8)).save(
            mrf_masks_dir / f"{img_id}_mrf_mask.png"
        )

        base_m = compute_binary_metrics(bm,       gt1, fv)
        mrf_m  = compute_binary_metrics(mrf_mask, gt1, fv)

        records.append({
            "image_id":            img_id,
            "baseline_accuracy":   base_m["accuracy"],
            "baseline_precision":  base_m["precision"],
            "baseline_recall":     base_m["recall_sensitivity"],
            "baseline_specificity":base_m["specificity"],
            "baseline_f1":         base_m["f1_score"],
            "mrf_initial_energy":  energy_hist[0],
            "mrf_final_energy":    energy_hist[-1],
            "mrf_accuracy":        mrf_m["accuracy"],
            "mrf_precision":       mrf_m["precision"],
            "mrf_recall":          mrf_m["recall_sensitivity"],
            "mrf_specificity":     mrf_m["specificity"],
            "mrf_f1":              mrf_m["f1_score"],
            "f1_improvement":      mrf_m["f1_score"] - base_m["f1_score"],
        })

    comp_df  = pd.DataFrame(records)
    csv_path = output_dir / "baseline_vs_mrf_comparison.csv"
    comp_df.to_csv(csv_path, index=False)

    print(f"\n[MRF Experiment] Results on {len(comp_df)} test images:")
    print(f"  Mean Baseline F1 : {comp_df['baseline_f1'].mean():.4f}")
    print(f"  Mean MRF F1      : {comp_df['mrf_f1'].mean():.4f}")
    print(f"  Mean Improvement : {comp_df['f1_improvement'].mean():+.4f}")
    return comp_df
