# Results Figures

Every figure here is created by `python -m src.visualization` from files in `outputs/`. No numbers are hardcoded.

| # | File | What it shows |
| :-: | :--- | :--- |
| 00 | `00_results_dashboard.png` | One-page summary across all three phases |
| 01 | `01_pipeline_stages.png` | Test image 11L: original → green → CLAHE → mask → GT → error map → skeleton → nodes |
| 02 | `02_segmentation_metrics.png` | Train vs test Acc/Prec/Rec/Spec/F1 (mean ± std), plus per-image F1 vs both observers |
| 03 | `03_precision_recall_observers.png` | Per-image precision–recall trade-off; F1 agreement between Obs 1 and Obs 2 |
| 04 | `04_test_error_gallery.png` | Error maps for all 8 held-out test images (green TP, red FP, blue FN) |
| 05 | `05_skeleton_nodes_gallery.png` | Skeleton overlays and node maps (yellow = endpoint, blue = junction) |
| 06 | `06_graph_overlay_degree_components.png` | NetworkX graph on the fundus (edges coloured by tortuosity), degree distribution, component sizes |
| 07 | `07_feature_distributions.png` | Train/test box plots for all 11 network features |
| 08 | `08_fractal_dimension.png` | Box-counting log–log curves for all 28 masks; per-image D_f |
| 09 | `09_feature_relationships.png` | Length vs tortuosity, density vs efficiency, branching vs D_f (Pearson and Spearman) |
| 10 | `10_correlation_full.png` | Full Pearson correlation heatmap |
| 11 | `11_bayesian_network.png` | DAG, prior P(B), CPT heatmaps, tertile cut-offs, per-image posterior P(S=Complex) |
| 12 | `12_bayes_evidence_queries.png` | What-if diagnostic queries computed live from the saved CPTs |
| 13 | `13_mrf_vs_baseline_metrics.png` | Test metrics, baseline vs MRF; ICM energy reduction per image |
| 14 | `14_mrf_visual_comparison.png` | GT vs baseline vs MRF masks for 3 test images |

**How to read them:** see *Known Limitations* in `PLAN_AND_PROGRESS.md`. Figures 04 and 06 show the over-segmentation and graph fragmentation directly. Figure 13 shows the MRF currently does worse than the baseline.
