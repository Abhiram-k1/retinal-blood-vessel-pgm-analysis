"""
Deep Learning / Neural Network Module for Retinal Vessel Segmentation.

Implements a Multi-Layer Convolutional/Receptive-Field Neural Network (CNN/MLP)
for supervised retinal blood-vessel segmentation, adhering to classical established
methodologies (e.g., Ricci & Ferrara 2007, Soares et al. 2006).

Architecture & Feature Representation:
  - 10-dimensional multi-scale spatial feature representation per pixel:
      1. Green-channel normalized intensity
      2. Local 5x5 spatial mean (receptive field context)
      3. Local 5x5 spatial standard deviation (texture energy)
      4. Multi-Scale Gabor response sigma=2 (fine capillaries)
      5. Multi-Scale Gabor response sigma=4 (medium vessels)
      6. Multi-Scale Gabor response sigma=6 (large vessels)
      7. Morphological bottom-hat curvilinear feature
      8. Spatial gradient magnitude (Sobel edge response)
      9. Gaussian blur sigma=2 (low-frequency context)
      10. High-pass Laplacian response (vessel ridge contrast)
  - Deep Multi-Layer Perceptron (MLP) with two hidden layers:
      Input (10) -> Dense (64, ReLU) -> Dense (32, ReLU) -> Output (2, Softmax / Probability)
  - Trained with Adam optimizer and cross-entropy loss on balanced training pixels.

Provides:
  - Continuous vessel probability map (for MRF unary potentials / calibration)
  - Binary vessel segmentation
  - Deep feature embeddings for vascular analysis
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.ndimage import uniform_filter, generic_gradient_magnitude, sobel, gaussian_laplace
from sklearn.neural_network import MLPClassifier
import joblib

from src.multiscale_features import compute_multiscale_gabor_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"


def extract_pixel_feature_stack(preproc_uint8: np.ndarray, fov_mask: np.ndarray) -> np.ndarray:
    """
    Extracts 10-dimensional multi-scale receptive field features for every pixel.
    
    Returns:
        feature_stack: float32 array of shape (H, W, 10)
    """
    img = preproc_uint8.astype(np.float32) / 255.0
    H, W = img.shape
    
    # 1. Base intensity
    f1 = img
    
    # 2. Local 5x5 spatial mean
    f2 = uniform_filter(img, size=5)
    
    # 3. Local 5x5 spatial standard deviation
    f3 = np.sqrt(np.maximum(0, uniform_filter(img**2, size=5) - f2**2))
    
    # 4, 5, 6. Multi-Scale Gabor responses (sigma=2, 4, 6)
    gabor_res = compute_multiscale_gabor_response(preproc_uint8, sigmas=[2, 4, 6])
    scale_resps = gabor_res["scale_responses"]
    f4 = scale_resps[0] / (np.max(scale_resps[0]) + 1e-6)
    f5 = scale_resps[1] / (np.max(scale_resps[1]) + 1e-6)
    f6 = scale_resps[2] / (np.max(scale_resps[2]) + 1e-6)
    
    # 7. Morphological bottom-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closing = cv2.morphologyEx(preproc_uint8, cv2.MORPH_CLOSE, kernel)
    f7 = (cv2.subtract(closing, preproc_uint8).astype(np.float32)) / 255.0
    
    # 8. Gradient magnitude
    sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    f8 = np.sqrt(sx**2 + sy**2)
    
    # 9. Gaussian context (sigma=2)
    f9 = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    
    # 10. Laplacian ridge response
    f10 = np.abs(cv2.Laplacian(img, cv2.CV_32F, ksize=3))
    
    stack = np.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], axis=-1)
    return stack.astype(np.float32)


class RetinalDeepSegmentationModel:
    """
    Supervised Deep Neural Network for Retinal Blood-Vessel Segmentation.
    """
    def __init__(self, hidden_layer_sizes: tuple = (64, 32), max_iter: int = 25, random_state: int = 42):
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=3,
            validation_fraction=0.15,
            batch_size=256
        )
        self.is_trained = False

    def train_on_manifest(self, manifest_df: pd.DataFrame, preproc_dir: Path, samples_per_image: int = 2500) -> None:
        """
        Trains deep model on the 20 CHASE_DB1 training images using balanced pixel sampling.
        """
        train_df = manifest_df[manifest_df["split"] == "train"]
        X_list, y_list = [], []
        rng = np.random.RandomState(42)

        print(f"[Deep Learning] Extracting training dataset across {len(train_df)} images ({samples_per_image} px/img)...")
        for _, row in train_df.iterrows():
            img_id = row["image_id"]
            pp_path = preproc_dir / f"{img_id}_preprocessed.png"
            preproc_uint8 = np.array(Image.open(pp_path))
            
            fov_mask = np.array(Image.open(row["mask_path"])) > 128
            if fov_mask.ndim == 3: fov_mask = fov_mask[:, :, 0]
            
            gt_mask = np.array(Image.open(row["manual1_path"])) > 128
            if gt_mask.ndim == 3: gt_mask = gt_mask[:, :, 0]
            
            feats = extract_pixel_feature_stack(preproc_uint8, fov_mask)
            
            vessel_idx = np.where(fov_mask & gt_mask)
            bg_idx = np.where(fov_mask & (~gt_mask))
            
            n_vessel = len(vessel_idx[0])
            n_bg = len(bg_idx[0])
            
            n_sample = min(samples_per_image // 2, n_vessel, n_bg)
            if n_sample > 0:
                v_sel = rng.choice(n_vessel, n_sample, replace=False)
                b_sel = rng.choice(n_bg, n_sample, replace=False)
                
                v_feats = feats[vessel_idx[0][v_sel], vessel_idx[1][v_sel]]
                b_feats = feats[bg_idx[0][b_sel], bg_idx[1][b_sel]]
                
                X_list.append(v_feats)
                y_list.append(np.ones(n_sample, dtype=np.int32))
                X_list.append(b_feats)
                y_list.append(np.zeros(n_sample, dtype=np.int32))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print(f"[Deep Learning] Training MLP Neural Network on {len(X)} balanced samples...")
        self.mlp.fit(X, y)
        self.is_trained = True
        print(f"[Deep Learning] Training complete! Training loss: {self.mlp.loss_:.4f}")

    def predict_image(self, preproc_uint8: np.ndarray, fov_mask: np.ndarray, threshold: float = 0.5) -> tuple:
        """
        Generates probability map and binary mask for a full retinal image.
        
        Returns:
            (prob_map, binary_mask)
        """
        feats = extract_pixel_feature_stack(preproc_uint8, fov_mask)
        H, W, D = feats.shape
        fov_flat = fov_mask.ravel()
        
        prob_map = np.zeros((H, W), dtype=np.float32)
        valid_feats = feats.reshape(-1, D)[fov_flat]
        
        if len(valid_feats) > 0:
            probs = self.mlp.predict_proba(valid_feats)[:, 1]
            prob_map.reshape(-1)[fov_flat] = probs
            
        binary_mask = ((prob_map > threshold) & fov_mask).astype(np.uint8)
        return prob_map, binary_mask

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.mlp, path)

    def load(self, path: Path) -> None:
        self.mlp = joblib.load(path)
        self.is_trained = True


def train_and_evaluate_deep_model(manifest_df: pd.DataFrame, preproc_dir: Path, output_dir: Path = OUTPUTS_DIR) -> tuple:
    """
    Trains deep learning model on training partition and generates predictions on test partition.
    """
    model = RetinalDeepSegmentationModel()
    model.train_on_manifest(manifest_df, preproc_dir)
    
    dl_masks_dir = output_dir / "dl_masks"
    dl_probs_dir = output_dir / "dl_probs"
    dl_masks_dir.mkdir(parents=True, exist_ok=True)
    dl_probs_dir.mkdir(parents=True, exist_ok=True)
    
    from src.validation import compute_binary_metrics
    records = []
    
    test_df = manifest_df[manifest_df["split"] == "test"]
    for _, row in test_df.iterrows():
        img_id = row["image_id"]
        pp = np.array(Image.open(preproc_dir / f"{img_id}_preprocessed.png"))
        fov = np.array(Image.open(row["mask_path"])) > 128
        if fov.ndim == 3: fov = fov[:, :, 0]
        gt1 = np.array(Image.open(row["manual1_path"])) > 128
        if gt1.ndim == 3: gt1 = gt1[:, :, 0]
        
        prob_map, bin_mask = model.predict_image(pp, fov)
        
        # Save artifacts
        Image.fromarray((bin_mask * 255).astype(np.uint8)).save(dl_masks_dir / f"{img_id}_dl_mask.png")
        prob_uint8 = (prob_map * 255.0).astype(np.uint8)
        Image.fromarray(prob_uint8).save(dl_probs_dir / f"{img_id}_dl_prob.png")
        
        m = compute_binary_metrics(bin_mask, gt1, fov)
        records.append({
            "image_id": img_id,
            "dl_accuracy": m["accuracy"],
            "dl_precision": m["precision"],
            "dl_recall": m["recall_sensitivity"],
            "dl_specificity": m["specificity"],
            "dl_f1": m["f1_score"],
            "dl_dice": m["dice_coefficient"],
            "dl_iou": m["iou"]
        })
        
    res_df = pd.DataFrame(records)
    res_df.to_csv(output_dir / "deep_learning_metrics.csv", index=False)
    print(f"\n[Deep Learning] Evaluation on Test Set ({len(res_df)} images):")
    print(f"  Mean DL F1 / Dice: {res_df['dl_f1'].mean():.4f}")
    print(f"  Mean DL Precision: {res_df['dl_precision'].mean():.4f}")
    print(f"  Mean DL Recall:    {res_df['dl_recall'].mean():.4f}")
    print(f"  Mean DL IoU:       {res_df['dl_iou'].mean():.4f}")
    
    # Save trained model
    model.save(output_dir / "reports" / "retinal_deep_model.joblib")
    return model, res_df

