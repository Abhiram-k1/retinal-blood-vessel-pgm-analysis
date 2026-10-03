"""
Results Visualization Module.
Renders figures for every pipeline stage strictly from files produced by src.pipeline
(no hardcoded or synthetic numbers). Output: outputs/figures/*.png

Run:  python -m src.visualization
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.graph_construction import build_vessel_graph
from src.fractal_analysis import compute_box_counting_fractal_dimension

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "outputs"
FIG = OUT / "figures"

TRAIN_C, TEST_C = "#2b6cb0", "#dd6b20"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _load_bool(path, thr=128):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[:, :, 0]
    return a > thr if a.dtype != bool else a


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {name}")


# ----------------------------------------------------------------------------- Phase 1
def fig_pipeline_stages(row):
    iid = row.image_id
    rgb = np.array(Image.open(row.image_path))[:, :, :3]
    imgs = [
        (rgb, "1. Original fundus", None),
        (rgb[:, :, 1], "2. Green channel", "gray"),
        (np.array(Image.open(OUT / "preprocessed" / f"{iid}_preprocessed.png")), "3. CLAHE + smoothing", "gray"),
        (_load_bool(OUT / "masks" / f"{iid}_vessel_mask.png"), "4. Predicted vessel mask", "gray"),
        (_load_bool(row.manual1_path), "5. Ground truth (Obs 1)", "gray"),
        (np.array(Image.open(OUT / "difference_maps" / f"{iid}_diff_vs_obs1.png")), "6. Error map (TP/FP/FN)", None),
        (np.array(Image.open(OUT / "skeletons" / "overlays" / f"{iid}_skel_overlay.png")), "7. Skeleton overlay", None),
        (np.array(Image.open(OUT / "nodes" / "visualizations" / f"{iid}_nodes_vis.png")), "8. Endpoints & junctions", None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (im, title, cmap) in zip(axes.ravel(), imgs):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"End-to-end pipeline stages — {iid} ({row.split})", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.01, "Error map: green = TP, red = FP, blue = FN, black = TN, grey = outside FOV",
             ha="center", fontsize=11)
    _save(fig, "01_pipeline_stages.png")


def fig_segmentation_metrics(seg):
    metrics = ["accuracy", "precision", "recall", "specificity", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    # (a) mean metrics train vs test
    x = np.arange(len(metrics)); w = 0.38
    for i, (split, col) in enumerate([("train", TRAIN_C), ("test", TEST_C)]):
        sub = seg[seg.split == split]
        means = [sub[f"obs1_{m}"].mean() for m in metrics]
        stds = [sub[f"obs1_{m}"].std() for m in metrics]
        bars = axes[0].bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4, color=col,
                           label=f"{split} (n={len(sub)})")
        for b, v in zip(bars, means):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([m.capitalize() for m in metrics])
    axes[0].set_ylim(0, 1.12); axes[0].set_title("Segmentation metrics vs Observer 1 (mean ± std, inside FOV)")
    axes[0].legend()
    # (b) per-image F1 vs both observers
    idx = np.arange(len(seg))
    axes[1].plot(idx, seg.obs1_f1, "o-", label="F1 vs Observer 1", color="#2f855a")
    axes[1].plot(idx, seg.obs2_f1, "s--", label="F1 vs Observer 2", color="#805ad5")
    n_train = (seg.split == "train").sum()
    axes[1].axvspan(n_train - 0.5, len(seg) - 0.5, color=TEST_C, alpha=0.12, label="held-out test")
    axes[1].set_xticks(idx); axes[1].set_xticklabels(seg.image_id.str.replace("Image_", ""), rotation=90, fontsize=8)
    axes[1].set_ylim(0, 1); axes[1].set_title("Per-image F1 against both human observers"); axes[1].legend()
    _save(fig, "02_segmentation_metrics.png")


def fig_confusion_and_tradeoff(seg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for split, col in [("train", TRAIN_C), ("test", TEST_C)]:
        sub = seg[seg.split == split]
        axes[0].scatter(sub.obs1_recall, sub.obs1_precision, s=60, color=col, label=split, edgecolor="k")
    axes[0].set_xlabel("Recall / Sensitivity"); axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_title("Precision–recall trade-off per image (Obs 1)"); axes[0].legend()
    # Observer agreement scatter
    axes[1].scatter(seg.obs1_f1, seg.obs2_f1, c=[TRAIN_C if s == "train" else TEST_C for s in seg.split],
                    s=60, edgecolor="k")
    lim = [min(seg.obs1_f1.min(), seg.obs2_f1.min()) - 0.02, max(seg.obs1_f1.max(), seg.obs2_f1.max()) + 0.02]
    axes[1].plot(lim, lim, "k--", lw=1)
    axes[1].set_xlabel("F1 vs Observer 1"); axes[1].set_ylabel("F1 vs Observer 2")
    axes[1].set_title("Inter-observer consistency of our segmentation")
    _save(fig, "03_precision_recall_observers.png")


def fig_test_error_gallery(manifest):
    test = manifest[manifest.split == "test"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    seg = pd.read_csv(OUT / "segmentation_metrics.csv").set_index("image_id")
    for ax, (_, r) in zip(axes.ravel(), test.iterrows()):
        ax.imshow(Image.open(OUT / "difference_maps" / f"{r.image_id}_diff_vs_obs1.png"))
        ax.set_title(f"{r.image_id}  F1={seg.loc[r.image_id, 'obs1_f1']:.3f}", fontsize=12)
        ax.axis("off")
    fig.suptitle("Held-out test set error maps (green=TP, red=FP, blue=FN)", fontsize=16, fontweight="bold")
    _save(fig, "04_test_error_gallery.png")


def fig_skeleton_nodes_gallery(manifest):
    sample = manifest.iloc[[0, 5, 20, 25]]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for j, (_, r) in enumerate(sample.iterrows()):
        axes[0, j].imshow(Image.open(OUT / "skeletons" / "overlays" / f"{r.image_id}_skel_overlay.png"))
        axes[0, j].set_title(f"{r.image_id}: mask + skeleton"); axes[0, j].axis("off")
        nodes = pd.read_csv(OUT / "nodes" / f"{r.image_id}_nodes.csv")
        axes[1, j].imshow(Image.open(OUT / "nodes" / "visualizations" / f"{r.image_id}_nodes_vis.png"))
        ne = (nodes.node_type == "endpoint").sum(); nj = (nodes.node_type == "junction").sum()
        axes[1, j].set_title(f"endpoints={ne}, junctions={nj}"); axes[1, j].axis("off")
    fig.suptitle("Skeletonization and node detection", fontsize=16, fontweight="bold")
    _save(fig, "05_skeleton_nodes_gallery.png")


# ----------------------------------------------------------------------------- Phase 2
def fig_graph_overlay(row):
    iid = row.image_id
    skel = _load_bool(OUT / "skeletons" / f"{iid}_skeleton.png")
    nodes = pd.read_csv(OUT / "nodes" / f"{iid}_nodes.csv")
    G = build_vessel_graph(skel, nodes)
    pos = {n: d["pos"] for n, d in G.nodes(data=True)}
    rgb = np.array(Image.open(row.image_path))[:, :, :3]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))
    axes[0].imshow(rgb); axes[0].axis("off")
    tort = np.array([d["tortuosity"] for _, _, d in G.edges(data=True)])
    ec = nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color=np.clip(tort, 1, 1.5),
                                edge_cmap=plt.cm.plasma, width=1.6)
    jn = [n for n, d in G.nodes(data=True) if d["node_type"] == "junction"]
    en = [n for n, d in G.nodes(data=True) if d["node_type"] == "endpoint"]
    nx.draw_networkx_nodes(G, pos, nodelist=en, node_size=5, node_color="cyan", ax=axes[0])
    nx.draw_networkx_nodes(G, pos, nodelist=jn, node_size=12, node_color="red", ax=axes[0])
    if ec is not None:
        cb = fig.colorbar(ec, ax=axes[0], fraction=0.04); cb.set_label("edge tortuosity L/d")
    axes[0].set_title(f"Graph G=(V,E) on fundus — |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")

    # Degree distribution
    degs = np.array([d for _, d in G.degree()])
    vals, counts = np.unique(degs, return_counts=True)
    axes[1].bar(vals, counts, color="#3182ce")
    axes[1].set_yscale("log"); axes[1].set_xlabel("node degree k"); axes[1].set_ylabel("count (log)")
    axes[1].set_title("Degree distribution P(k)")

    # Component size distribution
    sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    axes[2].loglog(np.arange(1, len(sizes) + 1), sizes, "o-", ms=3, color="#c05621")
    axes[2].set_xlabel("component rank"); axes[2].set_ylabel("nodes in component")
    axes[2].set_title(f"Connected components: {len(sizes)} (largest = {sizes[0]} nodes)")
    fig.suptitle(f"Mathematical vessel graph — {iid}", fontsize=16, fontweight="bold")
    _save(fig, "06_graph_overlay_degree_components.png")


def fig_feature_distributions(feat):
    cols = ["nodes", "edges", "branches", "connected_components", "density", "total_length",
            "mean_length", "mean_angle", "mean_tortuosity", "fractal_dimension", "global_efficiency"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    for ax, c in zip(axes.ravel(), cols):
        data = [feat[feat.split == "train"][c], feat[feat.split == "test"][c]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55)
        for patch, col in zip(bp["boxes"], [TRAIN_C, TEST_C]):
            patch.set_facecolor(col); patch.set_alpha(0.55)
        for i, d in enumerate(data):
            ax.scatter(np.random.default_rng(0).normal(i + 1, 0.05, len(d)), d, s=14, color="k", zorder=3)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["train", "test"])
        ax.set_title(f"{c}\nmean={feat[c].mean():.4g} ± {feat[c].std():.3g}", fontsize=11)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Distribution of extracted network features (28 images)", fontsize=16, fontweight="bold")
    _save(fig, "07_feature_distributions.png")


def fig_fractal(manifest, feat):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.cm.viridis
    for i, (_, r) in enumerate(manifest.iterrows()):
        m = _load_bool(OUT / "masks" / f"{r.image_id}_vessel_mask.png")
        res = compute_box_counting_fractal_dimension(m)
        axes[0].plot(np.log(1 / np.array(res["box_sizes"])), np.log(res["box_counts"]),
                     "-o", ms=3, lw=1, color=cmap(i / len(manifest)), alpha=0.8)
    axes[0].set_xlabel(r"$\log(1/\epsilon)$"); axes[0].set_ylabel(r"$\log N(\epsilon)$")
    axes[0].set_title("Box-counting curves for all 28 vessel masks")
    order = feat.sort_values("fractal_dimension")
    axes[1].barh(order.image_id.str.replace("Image_", ""), order.fractal_dimension,
                 color=[TRAIN_C if s == "train" else TEST_C for s in order.split])
    axes[1].axvline(feat.fractal_dimension.mean(), color="k", ls="--",
                    label=f"mean D_f = {feat.fractal_dimension.mean():.3f}")
    axes[1].set_xlim(order.fractal_dimension.min() - 0.05, order.fractal_dimension.max() + 0.03)
    axes[1].set_xlabel("Fractal dimension D_f"); axes[1].set_title("Per-image fractal dimension (blue=train, orange=test)")
    axes[1].legend()
    _save(fig, "08_fractal_dimension.png")


def fig_relationships(feat):
    pairs = [("total_length", "mean_tortuosity", "Length vs Tortuosity"),
             ("density", "global_efficiency", "Density vs Efficiency"),
             ("branches", "fractal_dimension", "Branching vs Fractal Dimension")]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (x, y, t) in zip(axes, pairs):
        for split, col in [("train", TRAIN_C), ("test", TEST_C)]:
            s = feat[feat.split == split]
            ax.scatter(s[x], s[y], color=col, s=55, edgecolor="k", label=split)
        k, b = np.polyfit(feat[x], feat[y], 1)
        xs = np.linspace(feat[x].min(), feat[x].max(), 50)
        ax.plot(xs, k * xs + b, "k--", lw=1)
        r = feat[[x, y]].corr().iloc[0, 1]
        rho = feat[[x, y]].corr(method="spearman").iloc[0, 1]
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{t}\nPearson r={r:.3f}, Spearman ρ={rho:.3f}")
        ax.legend()
    fig.suptitle("Mandated feature relationships", fontsize=16, fontweight="bold")
    _save(fig, "09_feature_relationships.png")


def fig_correlation(feat):
    cols = ["nodes", "edges", "branches", "connected_components", "density", "total_length",
            "mean_length", "mean_angle", "mean_tortuosity", "fractal_dimension", "global_efficiency"]
    corr = feat[cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right"); ax.set_yticklabels(cols); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, fraction=0.046)
    ax.set_title("Pearson correlation of all network features", fontsize=14, fontweight="bold")
    _save(fig, "10_correlation_full.png")


def fig_bayesian(feat, bn_res):
    with open(OUT / "reports" / "bayesian_network_cpts.json") as f:
        bn = json.load(f)
    cpts = bn["cpts"]
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(2, 4)

    # (a) DAG
    ax = fig.add_subplot(gs[0, 0])
    D = nx.DiGraph([("B", "T"), ("B", "D"), ("D", "η"), ("T", "S"), ("D", "S"), ("η", "S")])
    p = {"B": (0, 2), "T": (-1, 1), "D": (1, 1), "η": (1.6, 0.2), "S": (0, 0)}
    nx.draw_networkx(D, p, ax=ax, node_size=2200, node_color=["#bee3f8", "#c6f6d5", "#c6f6d5", "#fefcbf", "#fed7d7"],
                     font_size=15, font_weight="bold", arrowsize=22, edgecolors="k")
    ax.set_title("Bayesian network DAG\nB=branching, T=tortuosity, D=density,\nη=efficiency, S=network state")
    ax.axis("off"); ax.margins(0.2)

    states3 = ["Low", "Medium", "High"]
    # (b) P(B)
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(states3, [cpts["P(B)"][s] for s in states3], color="#3182ce")
    ax.set_ylim(0, 1); ax.set_title("Prior P(B)")
    for i, s in enumerate(states3):
        ax.text(i, cpts["P(B)"][s] + 0.02, f"{cpts['P(B)'][s]:.2f}", ha="center")

    # (c,d,e) conditional CPT heatmaps
    for k, (key, title) in enumerate([("P(T|B)", "P(T | B)"), ("P(D|B)", "P(D | B)"), ("P(eta|D)", "P(η | D)")]):
        ax = fig.add_subplot(gs[0, 2] if k == 0 else gs[1, k - 1])
        mat = np.array([[cpts[key][par][ch] for ch in states3] for par in states3])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.55 else "black")
        ax.set_xticks(range(3)); ax.set_xticklabels(states3); ax.set_yticks(range(3)); ax.set_yticklabels(states3)
        ax.set_xlabel("child state"); ax.set_ylabel("parent state"); ax.set_title(title)

    # (f) discretization thresholds on train data
    ax = fig.add_subplot(gs[0, 3])
    col_map = {"B": "branches", "T": "mean_tortuosity", "D": "density", "eta": "global_efficiency"}
    for i, (v, c) in enumerate(col_map.items()):
        vals = feat[c]; lo, hi = bn["discretization_bins"][v]
        z = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        ax.scatter(z, np.full(len(z), i) + np.random.default_rng(i).normal(0, 0.06, len(z)),
                   c=[TRAIN_C if s == "train" else TEST_C for s in feat.split], s=20)
        for t in (lo, hi):
            ax.axvline  # noqa
            ax.plot([(t - vals.min()) / (vals.max() - vals.min() + 1e-12)] * 2, [i - 0.3, i + 0.3], "r-", lw=2)
    ax.set_yticks(range(4)); ax.set_yticklabels(list(col_map.values()))
    ax.set_xlabel("normalised value"); ax.set_title("Tertile cut-offs (red) learned on train only")

    # (g) posterior P(S=Complex) per image
    ax = fig.add_subplot(gs[1, 2:])
    colors = ["#e53e3e" if s == "Complex" else "#38a169" for s in bn_res.true_state]
    ax.bar(bn_res.image_id.str.replace("Image_", ""), bn_res.posterior_prob_complex, color=colors)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    ax.set_ylim(0, 1); ax.set_ylabel("P(S = Complex | B,T,D,η)")
    ax.tick_params(axis="x", rotation=90)
    n_train = (bn_res.split == "train").sum()
    ax.axvspan(n_train - 0.5, len(bn_res) - 0.5, color=TEST_C, alpha=0.12)
    ax.set_title("Posterior by exact variable elimination (bar colour = rule-derived label: red Complex, green Normal; "
                 "shaded = test)")
    fig.suptitle("Directed PGM: structure, learned CPTs and posterior inference", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save(fig, "11_bayesian_network.png")


def fig_bayes_evidence_queries():
    """What-if queries on the learned BN, recomputed live from the saved CPTs."""
    with open(OUT / "reports" / "bayesian_network_cpts.json") as f:
        bn = json.load(f)
    cpts, st = bn["cpts"], bn["states"]

    def joint(b, t, d, e, s):
        return (cpts["P(B)"][b] * cpts["P(T|B)"][b][t] * cpts["P(D|B)"][b][d] *
                cpts["P(eta|D)"][d][e] * cpts["P(S|T,D,eta)"][f"T={t},D={d},eta={e}"][s])

    def post(ev):
        acc = {s: 0.0 for s in st["S"]}
        for b in [ev["B"]] if "B" in ev else st["B"]:
            for t in [ev["T"]] if "T" in ev else st["T"]:
                for d in [ev["D"]] if "D" in ev else st["D"]:
                    for e in [ev["eta"]] if "eta" in ev else st["eta"]:
                        for s in st["S"]:
                            acc[s] += joint(b, t, d, e, s)
        z = sum(acc.values())
        return acc["Complex"] / z

    queries = [("no evidence", {}), ("B=High", {"B": "High"}), ("T=High", {"T": "High"}),
               ("D=High", {"D": "High"}), ("B=High,T=High", {"B": "High", "T": "High"}),
               ("B=High,D=High", {"B": "High", "D": "High"}), ("B=Low,T=Low", {"B": "Low", "T": "Low"}),
               ("all High", {"B": "High", "T": "High", "D": "High", "eta": "High"})]
    vals = [post(q) for _, q in queries]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.barh([n for n, _ in queries], vals, color=plt.cm.Reds(0.3 + 0.6 * np.array(vals)))
    for b_, v in zip(bars, vals):
        ax.text(v + 0.01, b_.get_y() + b_.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlim(0, 1); ax.set_xlabel("P(S = Complex | evidence)")
    ax.set_title("Diagnostic reasoning: posterior of network state under different evidence", fontweight="bold")
    ax.invert_yaxis()
    _save(fig, "12_bayes_evidence_queries.png")


# ----------------------------------------------------------------------------- Phase 3
def fig_mrf(manifest, comp):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    metrics = ["accuracy", "precision", "recall", "specificity", "f1"]
    x = np.arange(len(metrics)); w = 0.38
    b = [comp[f"baseline_{m}"].mean() for m in metrics]
    m = [comp[f"mrf_{m}"].mean() for m in metrics]
    for off, vals, lab, col in [(-w / 2, b, "Morphological baseline", "#4a5568"), (w / 2, m, "MRF (ICM)", "#9f7aea")]:
        bars = axes[0].bar(x + off, vals, w, label=lab, color=col)
        for bb, v in zip(bars, vals):
            axes[0].text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([s.capitalize() for s in metrics]); axes[0].set_ylim(0, 1.1)
    axes[0].set_title("Test set: baseline vs MRF (mean over 8 images)"); axes[0].legend()

    rel = (comp.mrf_initial_energy - comp.mrf_final_energy) / comp.mrf_initial_energy.abs() * 100
    axes[1].bar(comp.image_id.str.replace("Image_", ""), rel, color="#805ad5")
    axes[1].set_ylabel("energy reduction (%)"); axes[1].set_title("ICM energy reduction E(X⁰) → E(X*) per test image")
    _save(fig, "13_mrf_vs_baseline_metrics.png")

    # visual comparison
    test = manifest[manifest.split == "test"].iloc[:3]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, (_, r) in enumerate(test.iterrows()):
        for j, (img, t) in enumerate([
            (_load_bool(r.manual1_path), "Ground truth"),
            (_load_bool(OUT / "masks" / f"{r.image_id}_vessel_mask.png"), "Baseline"),
            (_load_bool(OUT / "mrf_masks" / f"{r.image_id}_mrf_mask.png"), "MRF (ICM)")]):
            axes[i, j].imshow(img, cmap="gray"); axes[i, j].axis("off")
            axes[i, j].set_title(f"{r.image_id} — {t}")
    fig.suptitle("Undirected MRF segmentation vs baseline (test images)", fontsize=16, fontweight="bold")
    _save(fig, "14_mrf_visual_comparison.png")


def fig_dashboard(seg, feat, comp, bn_res):
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    t = seg[seg.split == "test"]
    axes[0, 0].bar(["Acc", "Prec", "Rec", "Spec", "F1"],
                   [t.obs1_accuracy.mean(), t.obs1_precision.mean(), t.obs1_recall.mean(),
                    t.obs1_specificity.mean(), t.obs1_f1.mean()], color=TEST_C)
    axes[0, 0].set_ylim(0, 1); axes[0, 0].set_title("Phase 1 — test segmentation (Obs 1)")
    axes[0, 1].hist(feat.mean_tortuosity, bins=10, color="#d53f8c", edgecolor="k")
    axes[0, 1].set_title(f"Phase 2 — mean tortuosity (μ={feat.mean_tortuosity.mean():.3f})")
    axes[0, 2].hist(feat.fractal_dimension, bins=10, color="#38a169", edgecolor="k")
    axes[0, 2].set_title(f"Phase 2 — fractal dimension (μ={feat.fractal_dimension.mean():.3f})")
    axes[1, 0].scatter(feat.nodes, feat.edges, c=feat.connected_components, cmap="viridis", s=60, edgecolor="k")
    axes[1, 0].set_xlabel("|V|"); axes[1, 0].set_ylabel("|E|"); axes[1, 0].set_title("Phase 2 — graph size (colour = #components)")
    axes[1, 1].hist(bn_res.posterior_prob_complex, bins=10, color="#e53e3e", edgecolor="k")
    axes[1, 1].set_title("Phase 2 — BN posterior P(S=Complex)")
    axes[1, 2].bar(["Baseline F1", "MRF F1"], [comp.baseline_f1.mean(), comp.mrf_f1.mean()], color=["#4a5568", "#9f7aea"])
    axes[1, 2].set_ylim(0, 1); axes[1, 2].set_title("Phase 3 — MRF vs baseline (test)")
    fig.suptitle("Project results dashboard", fontsize=18, fontweight="bold")
    _save(fig, "00_results_dashboard.png")


def main():
    print("[Visualization] Rendering figures from pipeline outputs...")
    manifest = pd.read_csv(OUT / "dataset_manifest.csv")
    seg = pd.read_csv(OUT / "segmentation_metrics.csv")
    feat = pd.read_csv(OUT / "retinal_network_features.csv")
    comp = pd.read_csv(OUT / "baseline_vs_mrf_comparison.csv")
    bn_res = pd.read_csv(OUT / "bayesian_inference_results.csv")
    sample = manifest[manifest.split == "test"].iloc[0]

    fig_dashboard(seg, feat, comp, bn_res)
    fig_pipeline_stages(sample)
    fig_segmentation_metrics(seg)
    fig_confusion_and_tradeoff(seg)
    fig_test_error_gallery(manifest)
    fig_skeleton_nodes_gallery(manifest)
    fig_graph_overlay(sample)
    fig_feature_distributions(feat)
    fig_fractal(manifest, feat)
    fig_relationships(feat)
    fig_correlation(feat)
    fig_bayesian(feat, bn_res)
    fig_bayes_evidence_queries()
    fig_mrf(manifest, comp)
    print(f"[Visualization] Done -> {FIG}")


if __name__ == "__main__":
    main()
