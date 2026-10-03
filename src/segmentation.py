"""
Vessel Segmentation Engine for Retinal Fundus Images.
Implements reproducible morphological vessel enhancement, adaptive thresholding,
and topological cleanup.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "masks"


def segment_retinal_vessels(
    preproc_uint8: np.ndarray,
    fov_mask: np.ndarray,
    morph_kernel_size: int = 15,
    min_vessel_area: int = 40,
    hole_fill_size: int = 25
) -> np.ndarray:
    """
    Segments retinal blood vessels from preprocessed grayscale image.
    
    Args:
        preproc_uint8: uint8 image (H, W), vessels are dark structures on lighter background
        fov_mask: bool mask (H, W), True inside retinal area
        morph_kernel_size: disk radius/diameter for morphological bottom-hat
        min_vessel_area: minimum pixel area for valid vessel components
        hole_fill_size: maximum area of interior holes to fill
        
    Returns:
        binary_mask: uint8 array (H, W) where 1 indicates vessel, 0 indicates background
    """
    # 1. Morphological Bottom-Hat Transform (Closing - Original)
    # Isolates dark curvilinear structures (vessels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closing = cv2.morphologyEx(preproc_uint8, cv2.MORPH_CLOSE, kernel)
    vessel_enhanced = cv2.subtract(closing, preproc_uint8)
    
    # 2. Local Contrast Stretching inside FOV
    vessel_enhanced[~fov_mask] = 0
    enhanced_float = vessel_enhanced.astype(np.float32)
    max_val = np.max(enhanced_float[fov_mask]) if np.any(fov_mask) else 1.0
    if max_val > 0:
        enhanced_float = (enhanced_float / max_val) * 255.0
    enhanced_norm = enhanced_float.astype(np.uint8)
    
    # 3. Adaptive & Statistical Thresholding
    # Combine Otsu's global threshold with local mean threshold for fine capillaries
    valid_pixels = enhanced_norm[fov_mask]
    if len(valid_pixels) > 0:
        otsu_val, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_val = max(18, otsu_val * 0.75)
    else:
        threshold_val = 25
        
    binary_initial = (enhanced_norm > threshold_val) & fov_mask
    
    # 4. Morphological Cleanup (eliminate spurious speckles & fill holes)
    cleaned = remove_small_objects(binary_initial, min_size=min_vessel_area)
    cleaned = remove_small_holes(cleaned, area_threshold=hole_fill_size)
    
    # 5. Final binary mask (uint8, 0 or 1)
    binary_mask = (cleaned & fov_mask).astype(np.uint8)
    return binary_mask


def batch_segment_dataset(manifest_df, preproc_dir: Path, output_dir: Path = OUTPUTS_DIR) -> None:
    """
    Runs vessel segmentation on all images in manifest_df and saves binary masks.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in manifest_df.iterrows():
        img_id = row["image_id"]
        preproc_path = preproc_dir / f"{img_id}_preprocessed.png"
        preproc_img = np.array(Image.open(preproc_path))
        
        mask_np = np.array(Image.open(row["mask_path"]))
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        fov_mask = mask_np > 128
        
        binary_mask = segment_retinal_vessels(preproc_img, fov_mask)
        
        # Save as 8-bit image (0 and 255 for visual inspection)
        save_path = output_dir / f"{img_id}_vessel_mask.png"
        Image.fromarray(binary_mask * 255).save(save_path)
        
    print(f"[Segmentation] Generated binary masks for {len(manifest_df)} images -> {output_dir}")
