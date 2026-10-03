"""
Exact Bayesian Inference Engine for Retinal Vascular Networks.
Implements Variable Elimination to compute posterior distributions:
    P(Network State S | Observed Evidence)
Verifies probability axioms (sums to 1.0) and evaluates test partition.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.bayesian_network import RetinalBayesianNetwork

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def query_posterior_network_state(bn: RetinalBayesianNetwork, evidence: dict) -> dict:
    """
    Computes exact posterior probability distribution over Network State S given partial evidence.
    
    Args:
        bn: trained RetinalBayesianNetwork
        evidence: dict of observed variable states, e.g. {'T': 'High', 'B': 'High'}
        
    Returns:
        dict: {'Normal': float, 'Complex': float} summing to 1.0
    """
    unnorm = {"Normal": 0.0, "Complex": 0.0}
    
    # Iterate over all possible states of non-target variables
    for b in ([evidence["B"]] if "B" in evidence else bn.states["B"]):
        for t in ([evidence["T"]] if "T" in evidence else bn.states["T"]):
            for d in ([evidence["D"]] if "D" in evidence else bn.states["D"]):
                for eta in ([evidence["eta"]] if "eta" in evidence else bn.states["eta"]):
                    for s in bn.states["S"]:
                        joint_p = bn.get_joint_probability(b, t, d, eta, s)
                        unnorm[s] += joint_p
                        
    total = sum(unnorm.values())
    if total > 0:
        posterior = {s: float(prob / total) for s, prob in unnorm.items()}
    else:
        posterior = {s: 0.5 for s in bn.states["S"]}
        
    # Validation check: must sum to 1.0
    assert abs(sum(posterior.values()) - 1.0) < 1e-4, f"Posterior does not normalize to 1: {posterior}"
    return posterior


def evaluate_test_set_inference(bn: RetinalBayesianNetwork, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates Bayesian posterior inference on all images across the dataset.
    """
    disc_df = bn.discretize_dataframe(features_df)
    results = []
    
    for idx, row in disc_df.iterrows():
        img_id = row["image_id"]
        evidence = {
            "B": row["B"],
            "T": row["T"],
            "D": row["D"],
            "eta": row["eta"]
        }
        
        post = query_posterior_network_state(bn, evidence)
        
        results.append({
            "image_id": img_id,
            "split": row["split"],
            "observed_branching": row["B"],
            "observed_tortuosity": row["T"],
            "observed_density": row["D"],
            "observed_efficiency": row["eta"],
            "true_state": row["S"],
            "posterior_prob_normal": post["Normal"],
            "posterior_prob_complex": post["Complex"],
            "inferred_state": "Complex" if post["Complex"] > 0.5 else "Normal"
        })
        
    res_df = pd.DataFrame(results)
    csv_path = OUTPUTS_DIR / "bayesian_inference_results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"[Bayesian Inference] Inferred state posteriors for {len(res_df)} images -> {csv_path}")
    return res_df
