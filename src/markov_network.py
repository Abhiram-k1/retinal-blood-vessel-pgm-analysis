"""
Markov Random Field (MRF) Spatial Energy Formulation for Retinal Vessel Segmentation.

Undirected graphical model on a 4-connected pixel lattice:

    E(X) = sum_i D_i(X_i) + lambda * sum_{<i,j>} V_ij(X_i, X_j)

where:
    D_i(X_i)        — Unary / data potential:
                        D_i(c) = 0.5 * ((I_i - mu_c) / sigma_c)^2 + log(sigma_c)
                        derived from class-conditional Gaussian intensity likelihoods.
    V_ij(X_i, X_j)  — Pairwise / smoothness potential (Ising/Potts):
                        beta * [X_i != X_j]  (penalises label discontinuity)
    lambda           — Spatial regularisation weight (tuned on training set)
    beta             — Smoothness strength within pairwise term

Gibbs distribution:  P(X) = (1/Z) * exp(-E(X))
MAP labelling:       X* = argmin_X E(X)  solved via ICM (see mrf_inference.py)

BUG-001 FIX: fit_likelihoods() is now called from the experiment runner using
training images, so Gaussian parameters are DATA-DRIVEN rather than hardcoded.
In the bottom-hat preprocessed space, vessel pixels are DARK (low intensity) and
background pixels are BRIGHT (high intensity) — the opposite of the original
hardcoded defaults (mu_vessel=0.65, mu_bg=0.15).
"""

import numpy as np


class RetinalMRFEnergyModel:
    """
    Formulates MRF energy potentials on a 4-connected spatial lattice.

    Attributes:
        lambda_weight  — Regularisation weight on pairwise term (tunable)
        beta_smooth    — Ising smoothness coefficient within pairwise term
        mu_vessel      — Gaussian mean of vessel pixel intensities (fitted)
        sigma_vessel   — Gaussian std  of vessel pixel intensities (fitted)
        mu_bg          — Gaussian mean of background pixel intensities (fitted)
        sigma_bg       — Gaussian std  of background pixel intensities (fitted)
        is_fitted      — True once fit_likelihoods() has been called
    """

    def __init__(self, lambda_weight: float = 1.5, beta_smooth: float = 0.8):
        self.lambda_weight = lambda_weight
        self.beta_smooth   = beta_smooth

        # Sensible fallback defaults for bottom-hat preprocessed CHASE_DB1 images
        # (vessels are dark structures after bottom-hat + CLAHE inversion)
        # These will be overwritten by fit_likelihoods() when called properly.
        self.mu_vessel    = 0.30   # bottom-hat vessel pixels: dark, ~0.25-0.40
        self.sigma_vessel = 0.10
        self.mu_bg        = 0.75   # background pixels: bright, ~0.65-0.85
        self.sigma_bg     = 0.12
        self.is_fitted    = False

    # ------------------------------------------------------------------
    # Parameter estimation from training data
    # ------------------------------------------------------------------
    def fit_likelihoods(
        self,
        train_preprocs: list,
        train_masks:    list,
        fov_masks:      list
    ) -> None:
        """
        Estimates class-conditional Gaussian likelihood parameters from training images.

        Args:
            train_preprocs : list of float32 arrays in [0,1], preprocessed images
            train_masks    : list of uint8/bool arrays (1 = vessel, 0 = background)
            fov_masks      : list of bool arrays (True = inside retinal disc)

        Note:
            In the bottom-hat enhanced space, vessel pixels appear as LOCAL BRIGHT
            features RELATIVE to the smooth background (closing - original).
            After normalisation to [0,1] with FOV masking the actual distribution
            must be measured empirically — do not assume vessel brightness direction.
        """
        vessel_pixels_all = []
        bg_pixels_all     = []

        for preproc, mask, fov in zip(train_preprocs, train_masks, fov_masks):
            img = preproc.astype(np.float32)
            fov_b = fov.astype(bool)
            vessel_b = (mask > 0) & fov_b
            bg_b     = (mask == 0) & fov_b

            v_pix = img[vessel_b]
            b_pix = img[bg_b]

            if len(v_pix) > 100:
                vessel_pixels_all.append(v_pix)
            if len(b_pix) > 100:
                bg_pixels_all.append(b_pix)

        if vessel_pixels_all:
            all_v = np.concatenate(vessel_pixels_all)
            self.mu_vessel    = float(np.mean(all_v))
            self.sigma_vessel = float(max(1e-3, np.std(all_v)))

        if bg_pixels_all:
            all_b = np.concatenate(bg_pixels_all)
            self.mu_bg    = float(np.mean(all_b))
            self.sigma_bg = float(max(1e-3, np.std(all_b)))

        self.is_fitted = True
        print(f"[MRF] Likelihoods fitted: "
              f"vessel mu={self.mu_vessel:.4f} sigma={self.sigma_vessel:.4f} | "
              f"bg mu={self.mu_bg:.4f} sigma={self.sigma_bg:.4f}")

    # ------------------------------------------------------------------
    # Energy computation
    # ------------------------------------------------------------------
    def compute_unary_potentials(
        self,
        preproc_img: np.ndarray,
        fov_mask:    np.ndarray
    ) -> tuple:
        """
        Computes per-pixel unary energy for label 0 (background) and label 1 (vessel).

        D_i(c) = 0.5 * ((I_i - mu_c) / sigma_c)^2 + log(sigma_c)
        (negative log-likelihood of Gaussian class-conditional distribution)

        Returns:
            (d_bg, d_vessel) — float32 arrays of shape (H, W)
        """
        img = preproc_img.astype(np.float32)

        d_bg     = 0.5 * ((img - self.mu_bg)     / self.sigma_bg)    ** 2 + np.log(self.sigma_bg)
        d_vessel = 0.5 * ((img - self.mu_vessel)  / self.sigma_vessel) ** 2 + np.log(self.sigma_vessel)

        # Outside FOV: force background label (high vessel cost, zero bg cost)
        outside = ~fov_mask.astype(bool)
        d_vessel[outside] = 1e6
        d_bg[outside]     = 0.0

        return d_bg.astype(np.float32), d_vessel.astype(np.float32)

    def compute_total_energy(
        self,
        labels:   np.ndarray,
        d_bg:     np.ndarray,
        d_vessel: np.ndarray,
        fov_mask: np.ndarray
    ) -> float:
        """
        Evaluates total MRF energy E(X) for a given label configuration.

        E(X) = sum_i D_i(X_i) + lambda * beta * sum_{<i,j>} [X_i != X_j]
        """
        fov_b = fov_mask.astype(bool)

        # Unary term
        unary = np.where(labels == 1, d_vessel, d_bg)
        unary_energy = float(np.sum(unary[fov_b]))

        # Pairwise 4-connected term (horizontal + vertical neighbours inside FOV)
        lam = self.lambda_weight * self.beta_smooth
        lv  = (labels * fov_b).astype(np.int32)

        h_diff = ((lv[:, :-1] != lv[:, 1:]) &
                  fov_b[:, :-1] & fov_b[:, 1:])
        v_diff = ((lv[:-1, :] != lv[1:, :]) &
                  fov_b[:-1, :] & fov_b[1:, :])

        pairwise_energy = lam * float(np.sum(h_diff) + np.sum(v_diff))
        return unary_energy + pairwise_energy
