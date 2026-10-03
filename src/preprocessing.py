"""
Image Preprocessing Pipeline for Retinal Fundus Images.
- Green channel extraction (highest vessel-to-background contrast)
- Noise reduction (Gaussian / bilateral filtering)
- Contrast enhancement (CLAHE: Contrast-Limited Adaptive Histogram Equalization)
- Intensity normalization and Field of View (FOV) masking
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "preprocessed"


def preprocess_fundus_image(
    image_path: str,
    mask_path: str = None,
    clip_limit: float = 2.5,
    tile_grid_size: tuple = (8, 8),
    gaussian_sigma: float = 1.0
) -> dict:
    """
    Preprocesses a single retinal fundus image.
    
    Returns:
        dict containing:
            - 'original_rgb': uint8 numpy array (H, W, 3)
            - 'green_channel': uint8 numpy array (H, W)
            - 'preprocessed': float32 numpy array normalized to [0, 1]
            - 'preprocessed_uint8': uint8 numpy array [0, 255]
            - 'fov_mask': bool numpy array (H, W) where True is valid retinal area
    """
    # 1. Load image and ensure 3-channel RGB
    pil_img = Image.open(image_path)
    img_np = np.array(pil_img)
    if img_np.ndim == 2:
        img_rgb = np.stack([img_np]*3, axis=-1)
    elif img_np.shape[2] == 4:
        img_rgb = img_np[:, :, :3]
    else:
        img_rgb = img_np
        
    # 2. Extract Green Channel
    green_ch = img_rgb[:, :, 1]
    
    # 3. Load or compute FOV mask
    if mask_path and os.path.exists(mask_path):
        mask_np = np.array(Image.open(mask_path))
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        fov_mask = mask_np > 128
    else:
        # Derive circular FOV mask automatically if none provided
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        fov_mask = gray > 15
        
    # 4. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_out = clahe.apply(green_ch)
    
    # 5. Gaussian smoothing for high-frequency noise attenuation
    blurred = cv2.GaussianBlur(clahe_out, (0, 0), sigmaX=gaussian_sigma)
    
    # 6. Apply FOV mask and Min-Max Normalization
    preproc_float = blurred.astype(np.float32) / 255.0
    preproc_float[~fov_mask] = 0.0
    
    preproc_uint8 = (preproc_float * 255.0).astype(np.uint8)
    
    return {
        "original_rgb": img_rgb,
        "green_channel": green_ch,
        "preprocessed": preproc_float,
        "preprocessed_uint8": preproc_uint8,
        "fov_mask": fov_mask
    }


def batch_preprocess_dataset(manifest_df, output_dir: Path = OUTPUTS_DIR) -> None:
    """
    Processes all images listed in manifest_df and saves preprocessed results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        res = preprocess_fundus_image(row["image_path"], row["mask_path"])
        
        save_path = output_dir / f"{img_id}_preprocessed.png"
        Image.fromarray(res["preprocessed_uint8"]).save(save_path)
        
    print(f"[Preprocessing] Completed {len(manifest_df)} images -> {output_dir}")
