"""
Statistical Analysis and Correlation Module for Retinal Vascular Features.
Computes descriptive statistics (mean, std, median, min, max) and evaluates
Pearson and Spearman correlation matrices across structural and geometric features.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def generate_feature_statistics(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes parametric and non-parametric summary statistics for all continuous features.
    """
    numeric_cols = [
        "nodes", "edges", "branches", "connected_components",
        "density", "total_length", "mean_length", "mean_angle",
        "mean_tortuosity", "fractal_dimension", "global_efficiency"
    ]
    
    summary = features_df[numeric_cols].describe().T
    summary["variance"] = features_df[numeric_cols].var()
    summary["median"] = features_df[numeric_cols].median()
    
    reordered_cols = ["count", "mean", "std", "variance", "min", "25%", "median", "75%", "max"]
    summary = summary[reordered_cols]
    
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "feature_summary_statistics.csv"
    summary.to_csv(summary_path)
    print(f"[Statistics] Generated summary statistics -> {summary_path}")
    return summary


def generate_correlation_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix and plots heatmap.
    Focuses on mandated relationships:
        - Length vs Tortuosity
        - Density vs Efficiency
        - Branching vs Fractal Dimension
    """
    numeric_cols = [
        "nodes", "edges", "branches", "density",
        "total_length", "mean_tortuosity", "fractal_dimension", "global_efficiency"
    ]
    
    corr = features_df[numeric_cols].corr(method="pearson")
    reports_dir = OUTPUTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(reports_dir / "feature_correlation_matrix.csv")
    
    # Render heatmap plot
    plots_dir = OUTPUTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(numeric_cols, fontsize=9)
    
    # Annotate values inside cells
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.title("Retinal Vascular Feature Pearson Correlation Matrix", pad=20, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plot_path = plots_dir / "correlation_matrix.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"[Statistics] Saved correlation matrix and plot -> {plot_path}")
    return corr
