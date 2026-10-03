"""
Markov Random Field (MRF) Spatial Energy Formulation for Retinal Vessel Segmentation.
Defines:
    - Data / Unary Potential: D_i(X_i) = -log P(I_i | X_i)
    - Pairwise / Smoothness Potential: V_ij(X_i, X_j) = beta * [X_i != X_j]
    - Total Energy: E(X) = sum_i D_i(X_i) + lambda * sum_<i,j> V_ij(X_i, X_j)
    - Gibbs Distribution: P(X) = 1/Z * exp(-E(X))
"""

import numpy as np


class RetinalMRFEnergyModel:
    """
    Formulates MRF energy potentials on a 4-connected spatial lattice.
    """
    def __init__(self, lambda_weight: float = 1.2, beta_smooth: float = 1.0):
        self.lambda_weight = lambda_weight
        self.beta_smooth = beta_smooth
        # Empirical Gaussian parameters for vessel and background in preprocessed space
        self.mu_vessel = 0.65
        self.sigma_vessel = 0.18
        self.mu_bg = 0.15
        self.sigma_bg = 0.15

    def fit_likelihoods(self, train_preprocs: list, train_masks: list, fov_masks: list) -> None:
        """
        Estimates Gaussian likelihood parameters from training set.
        """
        vessel_pixels = []
        bg_pixels = []
        
        for preproc, mask, fov in zip(train_preprocs, train_masks, fov_masks):
            v_pix = preproc[fov & (mask > 0)]
            b_pix = preproc[fov & (mask == 0)]
            if len(v_pix) > 0:
                vessel_pixels.append(v_pix)
            if len(b_pix) > 0:
                bg_pixels.append(b_pix)
                
        if len(vessel_pixels) > 0:
            all_v = np.concatenate(vessel_pixels)
            self.mu_vessel = float(np.mean(all_v))
            self.sigma_vessel = float(max(1e-2, np.std(all_v)))
            
        if len(bg_pixels) > 0:
            all_b = np.concatenate(bg_pixels)
            self.mu_bg = float(np.mean(all_b))
            self.sigma_bg = float(max(1e-2, np.std(all_b)))

    def compute_unary_potentials(self, preproc_img: np.ndarray, fov_mask: np.ndarray) -> tuple:
        """
        Computes unary energy D_i(0) and D_i(1) for each pixel.
        D_i(c) = 0.5 * ((I_i - mu_c) / sigma_c)^2 + log(sigma_c)
        """
        img = preproc_img.astype(np.float32)
        
        d_bg = 0.5 * ((img - self.mu_bg) / self.sigma_bg)**2 + np.log(self.sigma_bg)
        d_vessel = 0.5 * ((img - self.mu_vessel) / self.sigma_vessel)**2 + np.log(self.sigma_vessel)
        
        # Outside FOV: heavily favor background
        d_vessel[~fov_mask] = 1e6
        d_bg[~fov_mask] = 0.0
        
        return d_bg, d_vessel

    def compute_total_energy(self, labels: np.ndarray, d_bg: np.ndarray, d_vessel: np.ndarray, fov_mask: np.ndarray) -> float:
        """
        Evaluates exact total energy E(X) of current configuration X.
        """
        # Unary energy
        unary_energy = np.sum(np.where(labels == 1, d_vessel, d_bg)[fov_mask])
        
        # Pairwise 4-connected energy: horizontal and vertical differences
        labels_valid = labels * fov_mask
        h_diff = (labels_valid[:, :-1] != labels_valid[:, 1:]) & (fov_mask[:, :-1] & fov_mask[:, 1:])
        v_diff = (labels_valid[:-1, :] != labels_valid[1:, :]) & (fov_mask[:-1, :] & fov_mask[1:, :])
        
        pairwise_energy = self.lambda_weight * self.beta_smooth * (np.sum(h_diff) + np.sum(v_diff))
        total_energy = float(unary_energy + pairwise_energy)
        return total_energy
