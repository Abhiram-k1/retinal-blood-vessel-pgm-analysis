"""
Skeletonization Module for Retinal Vessel Networks.
Converts binary vessel regions into topological 1-pixel wide centerlines.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "skeletons"


def extract_vessel_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Computes 1-pixel wide morphological skeleton from binary vessel mask.
    
    Args:
        binary_mask: bool or uint8 array (H, W) where True/1 indicates vessel
        
    Returns:
        skeleton: uint8 array (H, W) where 1 indicates centerline pixel, 0 background
    """
    mask_bool = binary_mask > 0
    skel_bool = skeletonize(mask_bool, method="lee")
    return skel_bool.astype(np.uint8)


def create_skeleton_overlay(binary_mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """
    Creates visual overlay:
      - Blue/Teal: Full segmented vessel body
      - Yellow/Red: Extracted 1-pixel centerline skeleton
    """
    h, w = binary_mask.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    mask_b = binary_mask > 0
    skel_b = skeleton > 0
    
    overlay[mask_b] = [0, 150, 220]    # Vessel body: Cyan/Blue
    overlay[skel_b] = [255, 50, 50]    # Skeleton: Red
    
    return overlay


def batch_skeletonize_dataset(manifest_df, masks_dir: Path, output_dir: Path = OUTPUTS_DIR) -> None:
    """
    Processes all binary masks in masks_dir and saves skeletons and visual overlays.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        mask_path = masks_dir / f"{img_id}_vessel_mask.png"
        mask = np.array(Image.open(mask_path)) > 128
        
        skel = extract_vessel_skeleton(mask)
        
        # Save binary skeleton (0 and 255)
        Image.fromarray(skel * 255).save(output_dir / f"{img_id}_skeleton.png")
        
        # Save overlay
        overlay = create_skeleton_overlay(mask, skel)
        Image.fromarray(overlay).save(overlay_dir / f"{img_id}_skel_overlay.png")
        
    print(f"[Skeletonization] Extracted centerlines for {len(manifest_df)} images -> {output_dir}")
