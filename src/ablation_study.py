"""
Comprehensive Ablation Study Module for Retinal Vessel Analysis.

Systematically evaluates and isolates the individual and joint contributions of:
  Condition A: Morphological Baseline (Bottom-Hat + Adaptive Thresholding)
  Condition B: Multiresolution Analysis (MRA via Multi-Scale Gabor Filterbank)
  Condition C: MRF Spatial Regularization (ICM Energy Minimization on baseline)
  Condition D: Deep Learning (Supervised Neural Network with Receptive Fields)
  Condition E: Full Integrated System (Deep Learning + MRF Spatial Regularization)

All evaluations are performed strictly on the held-out test split (8 images, Patients 11-14)
evaluated inside the true Field of View (FOV) against human Observer 1.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.validation import compute_binary_metrics
from src.multiscale_features import enhance_segmentation_with_gabor
from src.markov_network import RetinalMRFEnergyModel
from src.mrf_inference import run_icm_inference
from src.deep_learning import RetinalDeepSegmentationModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


def run_comprehensive_ablation_study(manifest_df: pd.DataFrame, dl_model: RetinalDeepSegmentationModel = None) -> pd.DataFrame:
    """
    Executes the 5-condition ablation experiment on the test set.
    """
    preproc_dir = OUTPUTS_DIR / "preprocessed"
    masks_dir   = OUTPUTS_DIR / "masks"
    test_df     = manifest_df[manifest_df["split"] == "test"].copy()
    train_df    = manifest_df[manifest_df["split"] == "train"].copy()

    # Pre-train / load DL model if not provided
    if dl_model is None:
        dl_model = RetinalDeepSegmentationModel()
        model_path = OUTPUTS_DIR / "reports" / "retinal_deep_model.joblib"
        if model_path.exists():
            dl_model.load(model_path)
        else:
            dl_model.train_on_manifest(manifest_df, preproc_dir)

    # Fit MRF likelihoods strictly on training set
    print("[Ablation Study] Fitting MRF likelihoods on training set...")
    train_preprocs, train_masks_list, train_fovs = [], [], []
    for _, row in train_df.iterrows():
        img_id = row["image_id"]
        pp = np.array(Image.open(preproc_dir / f"{img_id}_preprocessed.png")).astype(np.float32) / 255.0
        bm = (np.array(Image.open(masks_dir  / f"{img_id}_vessel_mask.png")) > 128).astype(np.uint8)
        fv = np.array(Image.open(row["mask_path"])) > 128
        if fv.ndim == 3: fv = fv[:, :, 0]
        train_preprocs.append(pp)
        train_masks_list.append(bm)
        train_fovs.append(fv)

    mrf_model = RetinalMRFEnergyModel(lambda_weight=1.0, beta_smooth=0.8)
    mrf_model.fit_likelihoods(train_preprocs, train_masks_list, train_fovs)

    records = []
    conditions = [
        "A_Morphological_Baseline",
        "B_MRA_MultiScale_Gabor",
        "C_MRF_Spatial_Refinement",
        "D_Deep_Learning_Model",
        "E_Full_System_DL_MRF"
    ]

    print(f"\n[Ablation Study] Benchmarking 5 conditions across {len(test_df)} test images...")
    for _, row in test_df.iterrows():
        img_id = row["image_id"]
        pp_uint8 = np.array(Image.open(preproc_dir / f"{img_id}_preprocessed.png"))
        pp_float = pp_uint8.astype(np.float32) / 255.0
        baseline  = (np.array(Image.open(masks_dir / f"{img_id}_vessel_mask.png")) > 128).astype(np.uint8)
        fov       = np.array(Image.open(row["mask_path"])) > 128
        if fov.ndim == 3: fov = fov[:, :, 0]
        gt1       = np.array(Image.open(row["manual1_path"])) > 128
        if gt1.ndim == 3: gt1 = gt1[:, :, 0]

        # Condition A: Baseline
        mask_A = baseline

        # Condition B: MRA Gabor Enhancement
        mask_B = enhance_segmentation_with_gabor(pp_uint8, baseline, fov)

        # Condition C: MRF Spatial Refinement
        mask_C, _ = run_icm_inference(pp_float, fov, baseline, mrf_model, max_iters=6)

        # Condition D: Deep Learning Model
        _, mask_D = dl_model.predict_image(pp_uint8, fov)

        # Condition E: Full System (Deep Learning + MRF)
        mask_E, _ = run_icm_inference(pp_float, fov, mask_D, mrf_model, max_iters=6)

        test_masks = [mask_A, mask_B, mask_C, mask_D, mask_E]
        for cond_name, mask in zip(conditions, test_masks):
            m = compute_binary_metrics(mask, gt1, fov)
            records.append({
                "image_id":    img_id,
                "condition":   cond_name,
                "accuracy":    m["accuracy"],
                "precision":   m["precision"],
                "recall":      m["recall_sensitivity"],
                "specificity": m["specificity"],
                "f1_score":    m["f1_score"],
                "dice":        m["dice_coefficient"],
                "iou":         m["iou"],
            })

    ablation_df = pd.DataFrame(records)
    csv_path = OUTPUTS_DIR / "ablation_study_results.csv"
    ablation_df.to_csv(csv_path, index=False)

    summary = ablation_df.groupby("condition")[["accuracy", "precision", "recall", "specificity", "f1_score", "dice", "iou"]].mean()
    summary_path = OUTPUTS_DIR / "ablation_study_summary.csv"
    summary.to_csv(summary_path)

    print("\n================================================================================")
    print("                    ABLATION STUDY QUANTITATIVE SUMMARY                         ")
    print("================================================================================")
    print(summary.to_string())
    print("================================================================================\n")
    return ablation_df

