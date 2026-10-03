"""
Segmentation Validation Module for Retinal Vessel Extraction.
Calculates Accuracy, Precision, Recall/Sensitivity, Specificity, and F1-score
against human observer ground-truth masks strictly within Field of View (FOV).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, fov_mask: np.ndarray) -> dict:
    """
    Computes confusion matrix and validation metrics strictly evaluated within fov_mask.
    """
    # Restrict to pixels inside FOV
    pred_valid = (pred_mask > 0)[fov_mask]
    gt_valid = (gt_mask > 0)[fov_mask]
    
    tp = np.sum(pred_valid & gt_valid)
    fp = np.sum(pred_valid & ~gt_valid)
    tn = np.sum(~pred_valid & ~gt_valid)
    fn = np.sum(~pred_valid & gt_valid)
    total = tp + fp + tn + fn
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1)
    }


def create_difference_visualization(pred_mask: np.ndarray, gt_mask: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    """
    Generates color-coded difference map:
      - Green: True Positive (vessel correctly found)
      - Red: False Positive (over-segmented background)
      - Blue: False Negative (missed vessel)
      - Black: True Negative (correct background)
      - Gray: Outside FOV
    """
    h, w = pred_mask.shape[:2]
    diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    pred_b = pred_mask > 0
    gt_b = gt_mask > 0
    
    diff_rgb[fov_mask & pred_b & gt_b] = [0, 230, 0]      # TP: Green
    diff_rgb[fov_mask & pred_b & ~gt_b] = [230, 30, 30]   # FP: Red
    diff_rgb[fov_mask & ~pred_b & gt_b] = [30, 120, 255]  # FN: Blue
    diff_rgb[~fov_mask] = [60, 60, 60]                     # Outside FOV: Dark Gray
    
    return diff_rgb


def validate_dataset_segmentation(manifest_df, masks_dir: Path, output_dir: Path = OUTPUTS_DIR) -> pd.DataFrame:
    """
    Runs quantitative validation across all images in manifest_df against both Observer 1 and Observer 2.
    """
    records = []
    diff_dir = output_dir / "difference_maps"
    diff_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        pred_path = masks_dir / f"{img_id}_vessel_mask.png"
        pred_mask = np.array(Image.open(pred_path)) > 128
        
        # Ground Truth 1
        gt1_mask = np.array(Image.open(row["manual1_path"])) > 128
        if gt1_mask.ndim == 3:
            gt1_mask = gt1_mask[:, :, 0]
            
        # Ground Truth 2
        gt2_mask = np.array(Image.open(row["manual2_path"])) > 0
        if gt2_mask.ndim == 3:
            gt2_mask = gt2_mask[:, :, 0]
            
        # FOV Mask
        fov_mask = np.array(Image.open(row["mask_path"])) > 128
        if fov_mask.ndim == 3:
            fov_mask = fov_mask[:, :, 0]
            
        m1_res = compute_binary_metrics(pred_mask, gt1_mask, fov_mask)
        m2_res = compute_binary_metrics(pred_mask, gt2_mask, fov_mask)
        
        # Save difference map vs Observer 1
        diff_img = create_difference_visualization(pred_mask, gt1_mask, fov_mask)
        Image.fromarray(diff_img).save(diff_dir / f"{img_id}_diff_vs_obs1.png")
        
        records.append({
            "image_id": img_id,
            "split": row["split"],
            "obs1_accuracy": m1_res["accuracy"],
            "obs1_precision": m1_res["precision"],
            "obs1_recall": m1_res["recall_sensitivity"],
            "obs1_specificity": m1_res["specificity"],
            "obs1_f1": m1_res["f1_score"],
            "obs2_accuracy": m2_res["accuracy"],
            "obs2_precision": m2_res["precision"],
            "obs2_recall": m2_res["recall_sensitivity"],
            "obs2_specificity": m2_res["specificity"],
            "obs2_f1": m2_res["f1_score"]
        })
        
    df_metrics = pd.DataFrame(records)
    csv_path = output_dir / "segmentation_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    
    print(f"[Validation] Metrics computed for {len(df_metrics)} images -> {csv_path}")
    print(f"  Train Mean F1 (Obs 1): {df_metrics[df_metrics.split=='train']['obs1_f1'].mean():.4f}")
    print(f"  Test  Mean F1 (Obs 1): {df_metrics[df_metrics.split=='test']['obs1_f1'].mean():.4f}")
    return df_metrics
