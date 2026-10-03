"""
Fractal Dimension Analysis Module for Retinal Vasculature.
Estimates vascular self-similarity and branching complexity using
the multi-scale box-counting method.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def compute_box_counting_fractal_dimension(
    binary_mask: np.ndarray,
    box_sizes: list = [2, 4, 8, 16, 32, 64, 128, 256],
    save_plot_path: Path = None
) -> dict:
    """
    Calculates the Minkowski-Bouligand (box-counting) fractal dimension D_f.
    
    Args:
        binary_mask: 2D boolean or uint8 array where vessel pixels are True/1
        box_sizes: list of box scales epsilon (in pixels)
        save_plot_path: optional path to save log-log regression plot
        
    Returns:
        dict containing:
            - 'fractal_dimension': slope of log N(eps) vs log(1/eps)
            - 'r_squared': coefficient of determination of the linear fit
            - 'box_sizes': list of epsilons evaluated
            - 'box_counts': list of N(eps) counted
    """
    vessel_b = binary_mask > 0
    h, w = vessel_b.shape
    
    # Filter box sizes that fit within image dimensions
    valid_sizes = [s for s in box_sizes if s < min(h, w)]
    counts = []
    
    for s in valid_sizes:
        # Reshape or tile image into s x s blocks
        h_trim = (h // s) * s
        w_trim = (w // s) * s
        trimmed = vessel_b[:h_trim, :w_trim]
        
        # Reshape to (num_blocks_y, s, num_blocks_x, s)
        blocks = trimmed.reshape(h_trim // s, s, w_trim // s, s)
        # Any block with >= 1 vessel pixel is counted
        non_empty = np.any(blocks, axis=(1, 3))
        counts.append(int(np.sum(non_empty)))
        
    log_inv_eps = np.log(1.0 / np.array(valid_sizes, dtype=np.float64))
    log_counts = np.log(np.array(counts, dtype=np.float64))
    
    # Linear regression: log(N) = D_f * log(1/eps) + intercept
    poly_coeffs = np.polyfit(log_inv_eps, log_counts, 1)
    df = float(poly_coeffs[0])
    intercept = float(poly_coeffs[1])
    
    # Calculate R-squared
    fitted_vals = np.polyval(poly_coeffs, log_inv_eps)
    ss_tot = np.sum((log_counts - np.mean(log_counts))**2)
    ss_res = np.sum((log_counts - fitted_vals)**2)
    r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0
    
    if save_plot_path:
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4.5))
        plt.scatter(log_inv_eps, log_counts, color="#2b5c8f", s=40, label="Observed Box Counts")
        plt.plot(log_inv_eps, fitted_vals, color="#e63946", linestyle="--",
                 label=f"Fit: $D_f = {df:.4f}$ ($R^2 = {r_squared:.4f}$)")
        plt.xlabel(r"$\log(1 / \epsilon)$", fontsize=11)
        plt.ylabel(r"$\log N(\epsilon)$", fontsize=11)
        plt.title("Fractal Box-Counting Analysis", fontsize=12)
        plt.legend(frameon=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(save_plot_path, dpi=200)
        plt.close()
        
    return {
        "fractal_dimension": df,
        "r_squared": r_squared,
        "box_sizes": valid_sizes,
        "box_counts": counts
    }
