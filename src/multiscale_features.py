"""
Multi-Scale Gabor Filterbank Feature Extraction (Multiresolution Analysis component).

Gabor filters are directional bandpass filters that model the response of simple
cortical cells. For retinal vessel detection, a Gabor filterbank at multiple scales
and orientations captures curvilinear structures at different vessel widths:

    g(x, y; lambda, theta, psi, sigma, gamma) = 
        exp(-(x'^2 + gamma^2 * y'^2) / (2*sigma^2)) * cos(2*pi*x'/lambda + psi)

where x' = x*cos(theta) + y*sin(theta), y' = -x*sin(theta) + y*cos(theta).

Scales (sigma): [2, 4, 6] pixels — captures thin capillaries to large vessels
Orientations (theta): [0, 30, 60, 90, 120, 150] degrees — 6 directions
Wavelength (lambda): 2*sigma (matched to scale)
Aspect ratio (gamma): 0.5 (elongated filter tuned for vessels)
Phase offset (psi): 0 (cosine, symmetric)

The maximum response across all orientations at each scale gives a scale-specific
vessel enhancement map. The maximum across scales gives the multi-resolution
vessel probability map used to improve segmentation.

Reference:
    Chaudhuri et al. (1989) "Detection of Blood Vessels in Retinal Images Using
    Two-Dimensional Matched Filters." IEEE Trans Med Imaging 8(3):263-269.
    Jain & Farrokhnia (1991) "Unsupervised Texture Segmentation Using Gabor Filters."
    Pattern Recognition 24(12):1167-1186.
"""

import numpy as np
import cv2
from pathlib import Path


def build_gabor_kernel(
    ksize:  int,
    sigma:  float,
    theta:  float,
    lambd:  float,
    gamma:  float = 0.5,
    psi:    float = 0.0
) -> np.ndarray:
    """
    Constructs a single Gabor kernel using OpenCV.

    Args:
        ksize  : kernel size in pixels (must be odd)
        sigma  : Gaussian envelope standard deviation
        theta  : orientation in radians
        lambd  : wavelength of sinusoidal factor
        gamma  : spatial aspect ratio (0.5 for elongated filters)
        psi    : phase offset (0 for symmetric/cosine filter)

    Returns:
        kernel : float32 array of shape (ksize, ksize)
    """
    return cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi,
        ktype=cv2.CV_32F
    )


def compute_multiscale_gabor_response(
    image_uint8: np.ndarray,
    sigmas:      list = None,
    n_orientations: int = 6
) -> dict:
    """
    Computes multi-scale, multi-orientation Gabor filterbank responses.

    Args:
        image_uint8    : uint8 grayscale image (H, W) — preprocessed green channel
        sigmas         : list of Gaussian sigma values defining analysis scales
        n_orientations : number of evenly spaced orientations in [0, pi)

    Returns:
        dict containing:
            'scale_responses'    : list of (H,W) float32 max-over-orientation maps per scale
            'multiscale_max'     : (H,W) float32 maximum response across ALL scales and orientations
            'multiscale_mean'    : (H,W) float32 mean response across ALL scales and orientations
            'scale_labels'       : list of sigma values used
    """
    if sigmas is None:
        sigmas = [2, 4, 6]  # fine, medium, coarse vessel scales

    orientations = [np.pi * k / n_orientations for k in range(n_orientations)]
    img = image_uint8.astype(np.float32)

    all_responses   = []   # shape: (n_scales * n_orientations, H, W)
    scale_responses = []   # shape: (n_scales, H, W) — max over orientations

    for sigma in sigmas:
        ksize = int(6 * sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        lambd = 2.0 * sigma  # wavelength matched to scale

        orient_responses = []
        for theta in orientations:
            kernel = build_gabor_kernel(ksize, sigma, theta, lambd)
            # Convolve; take absolute value to capture both bright and dark vessel contrast
            resp = cv2.filter2D(img, cv2.CV_32F, kernel)
            resp_abs = np.abs(resp)
            orient_responses.append(resp_abs)
            all_responses.append(resp_abs)

        # Maximum response across orientations at this scale
        scale_max = np.max(np.stack(orient_responses, axis=0), axis=0)
        scale_responses.append(scale_max)

    all_stack       = np.stack(all_responses, axis=0)   # (N_filters, H, W)
    multiscale_max  = np.max(all_stack,  axis=0)         # (H, W)
    multiscale_mean = np.mean(all_stack, axis=0)         # (H, W)

    return {
        "scale_responses":  scale_responses,
        "multiscale_max":   multiscale_max,
        "multiscale_mean":  multiscale_mean,
        "scale_labels":     sigmas,
    }


def enhance_segmentation_with_gabor(
    preproc_uint8: np.ndarray,
    baseline_mask: np.ndarray,
    fov_mask:      np.ndarray,
    sigmas:        list = None,
    response_percentile: float = 70.0
) -> np.ndarray:
    """
    Refines a baseline binary vessel mask using multi-scale Gabor responses.

    Strategy:
    1. Compute multi-scale Gabor max-response map on the preprocessed image.
    2. Normalise response to [0,1] within FOV.
    3. Threshold at the given percentile of FOV pixels to get Gabor vessel candidates.
    4. Fuse: final mask = baseline_mask OR (Gabor candidates AND dilated_baseline).
       — This adds Gabor-detected vessel pixels that connect to existing segments.
       — It does NOT blindly add all Gabor activations (avoids false positives).

    Args:
        preproc_uint8        : uint8 preprocessed image
        baseline_mask        : uint8 binary mask from morphological segmentation
        fov_mask             : bool FOV mask
        sigmas               : Gabor scales
        response_percentile  : percentile threshold for Gabor candidate detection

    Returns:
        enhanced_mask : uint8 binary mask (0/1)
    """
    if sigmas is None:
        sigmas = [2, 4, 6]

    gabor_result = compute_multiscale_gabor_response(preproc_uint8, sigmas=sigmas)
    resp_map     = gabor_result["multiscale_max"].astype(np.float32)

    # Normalise within FOV
    fov_b     = fov_mask.astype(bool)
    fov_vals  = resp_map[fov_b]
    if len(fov_vals) > 0 and fov_vals.max() > 0:
        resp_map[fov_b] = (resp_map[fov_b] - fov_vals.min()) / (fov_vals.max() - fov_vals.min() + 1e-6)

    # Gabor candidate mask
    threshold      = float(np.percentile(resp_map[fov_b], response_percentile)) if fov_b.any() else 0.5
    gabor_cands    = (resp_map > threshold) & fov_b

    # Dilate baseline to find neighbourhood for fusion
    kern           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    base_dilated   = cv2.dilate(baseline_mask.astype(np.uint8), kern) > 0

    # Fusion: add Gabor candidates only near existing vessel pixels
    fused = (baseline_mask > 0) | (gabor_cands & base_dilated)
    enhanced_mask  = (fused & fov_b).astype(np.uint8)
    return enhanced_mask


def batch_compute_gabor_features(manifest_df, preproc_dir: Path, output_dir: Path) -> None:
    """
    Computes and saves multi-scale Gabor response maps for all dataset images.
    Saves normalised max-response maps for visual inspection.
    """
    from PIL import Image as PILImage
    gabor_dir = output_dir / "gabor_responses"
    gabor_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest_df.iterrows():
        img_id = row["image_id"]
        pp_path = preproc_dir / f"{img_id}_preprocessed.png"
        if not pp_path.exists():
            continue
        preproc = np.array(PILImage.open(pp_path))
        result  = compute_multiscale_gabor_response(preproc)
        resp    = result["multiscale_max"]

        # Normalise to uint8 for saving
        rmin, rmax = resp.min(), resp.max()
        if rmax > rmin:
            resp_uint8 = ((resp - rmin) / (rmax - rmin) * 255).astype(np.uint8)
        else:
            resp_uint8 = np.zeros_like(preproc, dtype=np.uint8)
        PILImage.fromarray(resp_uint8).save(gabor_dir / f"{img_id}_gabor_max.png")

    print(f"[MRA] Gabor response maps saved -> {gabor_dir}")
