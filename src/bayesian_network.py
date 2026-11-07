"""
Bayesian Network Modeling for Retinal Vascular Networks.

DAG Structure (Directed Acyclic Graph):
    Branching (B) -> Tortuosity (T)
    Branching (B) -> Density (D)
    Density (D)   -> Efficiency (eta)
    {T, D, eta}   -> Network State (S)

Joint factorisation:
    P(B, T, D, eta, S) = P(B) * P(T|B) * P(D|B) * P(eta|D) * P(S|T,D,eta)

BUG-003 FIX — Non-circular target label S:
    S is now defined from fractal_dimension < training median → "Complex".
    Fractal dimension is an independent continuous measure of vascular space-filling
    density and is NOT one of the four parent variables {B, T, D, eta} in the DAG.
    This eliminates the tautological posterior that resulted from the old rule where
    S was a deterministic function of (B, T, D) — the same variables used as evidence.

Parameter Learning: Maximum Likelihood with Laplace (add-one) smoothing.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "reports"


class RetinalBayesianNetwork:
    """
    Tabular Bayesian Network for retinal vascular feature dependencies.

    Random variables:
        B  — Branching count (branches / junctions in skeleton graph)
        T  — Mean vessel tortuosity index (L / d)
        D  — Graph network density (2E / N(N-1))
        eta— Global network transport efficiency (Latora & Marchiori, 2001)
        S  — Latent Network State: Normal or Complex

    DAG edges encode clinical domain knowledge:
        B → T  : Higher branching → more short, curved terminal segments → higher tortuosity
        B → D  : More branches → denser graph
        D → eta: Denser connected graph → higher transport efficiency
        {T,D,eta} → S : Network state integrates tortuosity, density, efficiency signals
    """

    def __init__(self):
        self.discretization_bins = {}
        self.cpts = {}
        self.variables = ["B", "T", "D", "eta", "S"]
        self.states = {
            "B":   ["Low", "Medium", "High"],
            "T":   ["Low", "Medium", "High"],
            "D":   ["Low", "Medium", "High"],
            "eta": ["Low", "Medium", "High"],
            "S":   ["Normal", "Complex"]
        }

    def fit_discretization(self, train_df: pd.DataFrame) -> None:
        """
        Derives tertile cutoff boundaries strictly from training split.

        S label threshold: fractal_dimension < training_median → "Complex"
        (impoverished vascular space-filling); otherwise "Normal".
        This threshold is causally independent of DAG parents {B, T, D, eta}
        and eliminates the circular posterior problem (BUG-003 fix).
        """
        for var, col in [
            ("B",   "branches"),
            ("T",   "mean_tortuosity"),
            ("D",   "density"),
            ("eta", "global_efficiency")
        ]:
            vals = train_df[col].dropna().values
            t1 = float(np.percentile(vals, 33.33))
            t2 = float(np.percentile(vals, 66.67))
            self.discretization_bins[var] = (t1, t2)

        # Independent label threshold for S: fractal_dimension median
        fd_vals = train_df["fractal_dimension"].dropna().values
        self.discretization_bins["S_fd_threshold"] = float(np.median(fd_vals))
        print(f"[Bayesian Network] Discretization bins fitted. "
              f"S threshold (fractal_dim median) = {self.discretization_bins['S_fd_threshold']:.4f}")

    def discretize_sample(self, row: pd.Series) -> dict:
        """
        Converts continuous measurements of one image into discrete states.

        Network State S is derived from fractal_dimension (independent of DAG parents):
            fractal_dimension < S_fd_threshold → "Complex"
            fractal_dimension >= S_fd_threshold → "Normal"

        Clinical rationale: Lower fractal dimension signals a sparser, less space-filling
        vascular network, consistent with vessel rarefaction in hypertension and DR.
        """
        disc = {}
        mapping = [
            ("B",   "branches"),
            ("T",   "mean_tortuosity"),
            ("D",   "density"),
            ("eta", "global_efficiency")
        ]
        for var, col in mapping:
            val = float(row[col])
            t1, t2 = self.discretization_bins[var]
            if val <= t1:
                disc[var] = "Low"
            elif val <= t2:
                disc[var] = "Medium"
            else:
                disc[var] = "High"

        # Non-circular S label — based solely on fractal_dimension
        fd_threshold = self.discretization_bins.get("S_fd_threshold", 1.666)
        fd_val = float(row.get("fractal_dimension", fd_threshold))
        disc["S"] = "Complex" if fd_val < fd_threshold else "Normal"
        return disc

    def discretize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Discretizes all rows of a features dataframe."""
        records = []
        for _, row in df.iterrows():
            d = self.discretize_sample(row)
            d["image_id"] = row["image_id"]
            d["split"] = row["split"]
            records.append(d)
        return pd.DataFrame(records)

    def learn_parameters(self, disc_train_df: pd.DataFrame, alpha: float = 1.0) -> None:
        """
        Learns Conditional Probability Tables (CPTs) via Laplace-smoothed MLE.

        Laplace smoothing (alpha=1):
            P(X_i = k | Pa(X_i)) = (count(X_i=k, Pa(X_i)) + alpha)
                                    / (count(Pa(X_i)) + alpha * |states(X_i)|)

        This prevents zero probabilities when training data is sparse (N=20 images).
        """
        N = len(disc_train_df)

        # 1. P(B): Prior distribution over Branching
        p_B = {}
        for b_state in self.states["B"]:
            count = int(np.sum(disc_train_df["B"] == b_state))
            p_B[b_state] = float((count + alpha) / (N + alpha * len(self.states["B"])))
        self.cpts["P(B)"] = p_B

        # 2. P(T | B): Tortuosity conditioned on Branching
        p_T_given_B = {}
        for b_state in self.states["B"]:
            sub = disc_train_df[disc_train_df["B"] == b_state]
            denom = len(sub) + alpha * len(self.states["T"])
            p_T_given_B[b_state] = {}
            for t_state in self.states["T"]:
                count = int(np.sum(sub["T"] == t_state))
                p_T_given_B[b_state][t_state] = float((count + alpha) / denom)
        self.cpts["P(T|B)"] = p_T_given_B

        # 3. P(D | B): Density conditioned on Branching
        p_D_given_B = {}
        for b_state in self.states["B"]:
            sub = disc_train_df[disc_train_df["B"] == b_state]
            denom = len(sub) + alpha * len(self.states["D"])
            p_D_given_B[b_state] = {}
            for d_state in self.states["D"]:
                count = int(np.sum(sub["D"] == d_state))
                p_D_given_B[b_state][d_state] = float((count + alpha) / denom)
        self.cpts["P(D|B)"] = p_D_given_B

        # 4. P(eta | D): Efficiency conditioned on Density
        p_eta_given_D = {}
        for d_state in self.states["D"]:
            sub = disc_train_df[disc_train_df["D"] == d_state]
            denom = len(sub) + alpha * len(self.states["eta"])
            p_eta_given_D[d_state] = {}
            for eta_state in self.states["eta"]:
                count = int(np.sum(sub["eta"] == eta_state))
                p_eta_given_D[d_state][eta_state] = float((count + alpha) / denom)
        self.cpts["P(eta|D)"] = p_eta_given_D

        # 5. P(S | T, D, eta): Network State from topological signals
        p_S_given_parents = {}
        for t_state in self.states["T"]:
            for d_state in self.states["D"]:
                for eta_state in self.states["eta"]:
                    key = f"T={t_state},D={d_state},eta={eta_state}"
                    sub = disc_train_df[
                        (disc_train_df["T"] == t_state) &
                        (disc_train_df["D"] == d_state) &
                        (disc_train_df["eta"] == eta_state)
                    ]
                    denom = len(sub) + alpha * len(self.states["S"])
                    p_S_given_parents[key] = {}
                    for s_state in self.states["S"]:
                        count = int(np.sum(sub["S"] == s_state))
                        p_S_given_parents[key][s_state] = float((count + alpha) / denom)
        self.cpts["P(S|T,D,eta)"] = p_S_given_parents

        # Print class balance of S in training set
        s_counts = disc_train_df["S"].value_counts().to_dict()
        print(f"[Bayesian Network] Training S label distribution: {s_counts}")

    def get_joint_probability(self, b, t, d, eta, s) -> float:
        """
        Computes joint probability via Bayesian factorisation:
            P(B,T,D,eta,S) = P(B) * P(T|B) * P(D|B) * P(eta|D) * P(S|T,D,eta)
        """
        p1 = self.cpts["P(B)"][b]
        p2 = self.cpts["P(T|B)"][b][t]
        p3 = self.cpts["P(D|B)"][b][d]
        p4 = self.cpts["P(eta|D)"][d][eta]
        parent_key = f"T={t},D={d},eta={eta}"
        p5 = self.cpts["P(S|T,D,eta)"][parent_key][s]
        return float(p1 * p2 * p3 * p4 * p5)

    def verify_factorization_normalization(self) -> float:
        """
        Sums joint probability over all 3×3×3×3×2 = 162 state combinations.
        Must equal 1.0 within numerical precision.
        """
        total = 0.0
        for b in self.states["B"]:
            for t in self.states["T"]:
                for d in self.states["D"]:
                    for eta in self.states["eta"]:
                        for s in self.states["S"]:
                            total += self.get_joint_probability(b, t, d, eta, s)
        return total

    def save_model(self, output_path: Path) -> None:
        """Exports discretization bins and learned CPTs to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "discretization_bins": self.discretization_bins,
            "states": self.states,
            "cpts": self.cpts
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Bayesian Network] CPTs saved -> {output_path}")


def train_retinal_bayesian_network(features_df: pd.DataFrame) -> "RetinalBayesianNetwork":
    """
    Fits and validates Bayesian Network strictly on training partition.
    Returns the trained RetinalBayesianNetwork instance.
    """
    bn = RetinalBayesianNetwork()
    train_df = features_df[features_df["split"] == "train"].copy()

    bn.fit_discretization(train_df)
    disc_train = bn.discretize_dataframe(train_df)
    bn.learn_parameters(disc_train)

    norm_sum = bn.verify_factorization_normalization()
    print(f"[Bayesian Network] Factorization check: Sum = {norm_sum:.6f} (Target: 1.000000)")
    assert abs(norm_sum - 1.0) < 1e-4, f"Factorization does not normalise to 1: sum={norm_sum}"

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    bn.save_model(OUTPUTS_DIR / "bayesian_network_cpts.json")
    return bn
