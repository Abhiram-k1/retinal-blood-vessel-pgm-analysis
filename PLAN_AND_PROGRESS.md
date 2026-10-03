# Project Plan & Progress Tracker (PLAN_AND_PROGRESS.md)

**Project:** Probabilistic and Mathematical Analysis of Retinal Blood-Vessel Networks  
**Course Focus:** Probabilistic Graphical Models (PGM)  
**Last Updated:** 2026-10-03  
**Status Overview:**
* **Phase 1 Target:** 100% Completion (Complete Foundation, Data Pipeline, Segmentation, GT Validation, Skeletonization, Node Detection)
* **Phase 2 Target:** $\ge$85% Completion (Graph Modeling, Topology, Geometry, Fractals, Efficiency, Master Feature Table, Bayesian Network DAG, CPT Learning, Factorization, Variable Elimination Inference, Statistics & Correlations)
* **Phase 3 Target:** Advanced / Synthesis (~15% Remaining Scope: MRF Energy Formulation, ICM Optimization, Baseline vs MRF Benchmarking, Comparative Theory, Final Report & Presentation)

---

## 1. Overall Progress Dashboard

| Phase | Target Completion | Current Status | Progress % | Primary Artifacts |
| :--- | :---: | :---: | :---: | :--- |
| **Phase 1: Ingestion, Preprocessing, Segmentation, Validation, Skeletons** | **100%** | **Implemented** | **100%** | `dataset_manifest.csv`, 28 preprocessed images, 28 masks, `segmentation_metrics.csv`, 28 skeletons, 28 node maps, figures 01–05 |
| **Phase 2: Graph Theory, Geometry, Fractals, Efficiency & Bayesian Networks** | **$\ge$85%** | **Implemented (quality issues noted below)** | **~85%** | graph files (JSON & GraphML), `retinal_network_features.csv`, box-counting plots, learned CPTs JSON, inference engine, figures 06–12 |
| **Phase 3: MRF Energy Model, Inference Optimization, Comparative Synthesis & Report** | **Remaining scope** | **Implemented, not tuned** | **~50%** | MRF energy model, ICM engine, `baseline_vs_mrf_comparison.csv`, `final_retinal_pgm_results.csv`, figures 13–14 |

## 1a. Measured Results (from `outputs/`, regenerate with `python -m src.pipeline`)

| Metric | Train (20) | Test (8) |
| :--- | :---: | :---: |
| Accuracy vs Obs 1 | 0.879 | 0.890 |
| Precision vs Obs 1 | 0.462 | 0.446 |
| Recall vs Obs 1 | 0.783 | 0.782 |
| Specificity vs Obs 1 | 0.890 | 0.901 |
| F1 vs Obs 1 | 0.578 | 0.567 |
| F1 vs Obs 2 | 0.549 | 0.587 |

| Feature (all 28 images) | Mean ± Std |
| :--- | :---: |
| Nodes / Edges | 1498 ± 506 / 754 ± 252 |
| Connected components | 751 ± 259 |
| Mean tortuosity | 1.080 ± 0.013 |
| Mean branch angle | 121.0° ± 2.0° |
| Fractal dimension $D_f$ | 1.666 ± 0.049 |
| Global efficiency | 0.0002 ± 0.0001 |
| MRF test F1 vs baseline | 0.396 vs 0.567 (MRF is **worse**) |

### Known Limitations (to fix next)
1. **Over-segmentation:** precision ≈ 0.45, so false-positive speckle inflates node counts.
2. **Fragmented graphs:** components ≈ edges (≈ 750), so most edges are isolated segments. Global efficiency and density are therefore very low and degree-2 merging / gap bridging is needed.
3. **Branch angle (~121°)** comes from node-to-neighbour chord vectors over all incident pairs, not from parent/daughter branch vectors, so it is not comparable to Murray's ~75° yet.
4. **Bayesian network label is circular:** `S` is defined by a rule over `B`, `T`, `D`, so 100% agreement is by construction (and only 2 of 28 images are `Complex`). A clinical or independent label is needed for a meaningful evaluation.
5. **MRF is untuned:** the unary likelihood is fit on preprocessed intensity, where vessels are *darker*, but the default means assume vessels are brighter. `fit_likelihoods` is never called, which is why F1 drops. This needs fixing plus a λ/β sweep on train.

---

## 2. Granular Task Checklist & Sprint Tracking

### Phase 1: Foundation, Data Ingestion, Segmentation & Topology (Target: 100%)

- [x] **Sprint 1.1: Project Setup & Environment Configuration (Phase 0)**
  - [x] Initialize modular repository structure (`src/`, `data/`, `outputs/`, `notebooks/`).
  - [x] Create `.gitignore` for Python environments and checkpoints.
  - [x] Setup isolated virtual environment (`.venv`) with NumPy, SciPy, Pillow, NetworkX, scikit-image, scikit-learn, OpenCV, Matplotlib.
  - [x] Author comprehensive plain-English `pseudo.md` explaining What, Why, and How.

- [x] **Sprint 1.2: Dataset Ingestion & Patient-Level Splitting (Phase 1)**
  - [x] Ingest CHASE_DB1 retinal images (28 total images, $999 \times 960$ resolution).
  - [x] Map 1st manual expert annotations (`01_manual1.tif` – `20_manual1.tif` / `01_manual1.tif` – `08_manual1.tif`).
  - [x] Map 2nd manual expert annotations (`Image_01L_2ndHO.png` – `Image_14R_2ndHO.png`).
  - [x] Map Field of View (FOV) masks (`mask_01L.png` – `mask_10R.png` / `01_test.tif` – `08_test.tif`).
  - [x] Enforce patient-level train (20 images: 01L–10R) / test (8 images: 11L–14R) separation to prevent data leakage.
  - [x] Generate automated `data/dataset_manifest.csv`.

- [x] **Sprint 1.3: Preprocessing Pipeline (Phase 2)**
  - [x] Extract green channel ($I_G$) for optimal hemoglobin absorption contrast.
  - [x] Apply bilateral / Gaussian filtering for noise reduction while preserving edges.
  - [x] Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization) with clip limit 2.0.
  - [x] Min-max intensity normalization to $[0, 1]$.
  - [x] Save preprocessed images to `outputs/preprocessed/` and generate side-by-side comparison figures.

- [x] **Sprint 1.4: Vessel Segmentation Engine (Phase 3)**
  - [x] Implement morphological top-hat / Gabor matched filtering for curvilinear enhancement.
  - [x] Adaptive local thresholding within FOV mask boundaries.
  - [x] Morphological opening and closing for noise suppression and hole-filling.
  - [x] Generate and persist binary vessel masks ($1 = \text{vessel}, 0 = \text{background}$) for all 28 images into `outputs/masks/`.

- [x] **Sprint 1.5: Quantitative Segmentation Validation (Phase 4)**
  - [x] Compute pixel-wise confusion matrix inside FOV mask against 1st and 2nd manual annotations.
  - [x] Compute Accuracy, Precision, Recall / Sensitivity, Specificity, and F1-Score.
  - [x] Save complete validation log into `outputs/segmentation_metrics.csv`.
  - [x] Generate error difference maps (Green = True Positive, Red = False Positive, Blue = False Negative).

- [x] **Sprint 1.6: Skeletonization & Medial Axis Thinning (Phase 5)**
  - [x] Apply topological thinning (Zhang-Suen / Guo-Hall / Medial Axis) to obtain 1-pixel wide centerlines.
  - [x] Verify topological connectivity and prune minor noise spurs.
  - [x] Save skeletons into `outputs/skeletons/` and generate mask vs skeleton overlays.

- [x] **Sprint 1.7: Node & Branch Detection (Phase 6)**
  - [x] Compute 8-neighborhood cross-number on skeletons:
    - 1 neighbor = Endpoint.
    - 2 neighbors = Normal centerline pixel.
    - 3+ neighbors = Junction / bifurcation candidate.
  - [x] Cluster adjacent junction pixels to compute unique centroid nodes.
  - [x] Save annotated node maps and coordinate registries into `outputs/nodes/`.

---

### Phase 2: Mathematical Graph Theory, Geometry, Fractals & Bayesian Networks (Target: $\ge$85%)

- [x] **Sprint 2.1: Mathematical Graph Construction (Phase 7)**
  - [x] Build topological graph $G=(V, E)$ using `NetworkX`.
  - [x] Assign spatial node attributes $(x, y)$ and node types (endpoint vs junction).
  - [x] Trace vessel segments along skeleton paths to assign edge attributes: pixel sequence, curve length $L$, and Euclidean chord distance $d$.

- [x] **Sprint 2.2: Graph-Theoretic Analysis (Phase 8)**
  - [x] Compute total node count $N = |V|$ and edge count $E = |E|$.
  - [x] Compute node degree distributions and identify high-degree hub junctions.
  - [x] Identify number of connected components $C$ to measure network fragmentation.
  - [x] Compute network density $D = \frac{2E}{N(N-1)}$.
  - [x] Compute shortest paths between connected node pairs.

- [x] **Sprint 2.3: Vascular Geometry & Tortuosity Analysis (Phase 9)**
  - [x] Calculate true curve path length $L = \sum \sqrt{\Delta x^2 + \Delta y^2}$.
  - [x] Calculate straight-line Euclidean distance $d(u, v)$ with $d > 0$ safety guards.
  - [x] Compute vessel tortuosity index $T = L / d$.
  - [x] Compute bifurcation branch angles $\theta$ from incident branch vectors using dot products.

- [x] **Sprint 2.4: Fractal Dimension Analysis (Phase 10)**
  - [x] Implement multi-scale box-counting algorithm across box sizes $\epsilon \in \{2, 4, 8, 16, 32, 64, 128\}$.
  - [x] Count non-empty vessel boxes $N(\epsilon)$ for each box size.
  - [x] Perform linear regression of $\log N(\epsilon)$ on $\log(1/\epsilon)$ to estimate fractal dimension $D_f$.
  - [x] Save log-log box-counting plots with regression coefficients and $R^2$ fit scores.

- [x] **Sprint 2.5: Global Network Efficiency (Phase 11)**
  - [x] Calculate all-pairs shortest path distance matrix $d(i, j)$.
  - [x] Compute global efficiency: $E_{\text{global}} = \frac{1}{N(N-1)} \sum_{i \ne j} \frac{1}{d(i, j)}$.
  - [x] Strictly handle disconnected node pairs by setting $1/d(i, j) = 0$ (no arbitrary finite penalties).

- [x] **Sprint 2.6: Master Feature Table Compilation (Phase 12)**
  - [x] Generate master tabular summary `outputs/retinal_network_features.csv` covering all 28 images.
  - [x] Feature columns: `image_id`, `nodes`, `edges`, `branches`, `connected_components`, `density`, `total_length`, `mean_length`, `mean_angle`, `mean_tortuosity`, `fractal_dimension`, `global_efficiency`.
  - [x] Verify zero numerical fabrication—every metric computed from real processed images.

- [x] **Sprint 2.7: Bridging to PGM & Feature Discretization (Phase 13)**
  - [x] Define PGM random variables: Tortuosity ($T$), Branching ($B$), Density ($D$), Network Efficiency ($\eta$), Network State ($S$).
  - [x] Discretize continuous variables into `{Low, Medium, High}` based on empirical tertiles derived strictly from the 20 training images.

- [x] **Sprint 2.8: Bayesian Network DAG Design (Phase 14)**
  - [x] Define Directed Acyclic Graph topology capturing vascular dependencies:
    - $B \rightarrow T$ (Branching influences segment tortuosity).
    - $B \rightarrow D$ (Branching dictates network density).
    - $D \rightarrow \eta$ (Density influences global transport efficiency).
    - $\{T, D, \eta\} \rightarrow S$ (Tortuosity, density, and efficiency jointly determine network complexity state).
  - [x] Verify DAG acyclicity and visualize DAG structure.

- [x] **Sprint 2.9: Bayesian Parameter Learning & Factorization (Phases 15 & 16)**
  - [x] Estimate prior $P(B)$ and conditional probability tables $P(T \mid B)$, $P(D \mid B)$, $P(\eta \mid D)$, $P(S \mid T, D, \eta)$ using Laplace-smoothed Maximum Likelihood Estimation strictly from training set.
  - [x] Prove joint distribution factorization: $P(B, T, D, \eta, S) = P(B)P(T|B)P(D|B)P(\eta|D)P(S|T,D,\eta)$.
  - [x] Verify probability axioms: all distributions normalized to $\sum = 1.0$.

- [x] **Sprint 2.10: Exact Bayesian Inference Engine (Phase 17)**
  - [x] Implement exact Variable Elimination engine.
  - [x] Query posterior probabilities given observed clinical evidence, e.g., $P(S \mid T=\text{High}, B=\text{High})$.
  - [x] Verify posterior distributions sum to 1.0 across test cases.

- [x] **Sprint 2.11: Statistical Summaries & Correlation Analysis (Phase 23)**
  - [x] Compute descriptive statistics (mean, median, standard deviation, variance, min, max) for all features.
  - [x] Generate Pearson and Spearman correlation matrices:
    - Length vs Tortuosity.
    - Density vs Efficiency.
    - Branching vs Fractal Dimension.
  - [x] Generate correlation heatmaps and bivariate scatter plots.

---

### Phase 3: MRF Spatial Modeling, Comparative Analysis & Final Synthesis (Remaining ~15% Scope)

- [ ] **Sprint 3.1: Undirected Markov Random Field Formulation (Phases 18 & 19)**
  - [x] Scaffold 4-connected / 8-connected lattice MRF model on pixel regions.
  - [x] Define unary/data term $D_i(X_i) = -\log P(I_i \mid X_i)$ from intensity likelihood.
  - [x] Define pairwise smoothness potential $V_{ij}(X_i, X_j) = \beta [X_i \ne X_j]$ (Ising/Potts prior).
  - [ ] Full parameter tuning of $\lambda$ and $\beta$ hyperparameters.

- [ ] **Sprint 3.2: MRF Inference & Optimization (Phase 20)**
  - [x] Scaffold Iterated Conditional Modes (ICM) local energy minimization.
  - [ ] Run full convergence benchmarks on test set.

- [ ] **Sprint 3.3: Controlled Baseline vs MRF Experiment (Phase 21)**
  - [ ] Quantify metric improvements (Accuracy, Precision, Recall, F1, Specificity) of MRF vs morphological baseline on held-out test set.
  - [ ] Generate comparative difference masks.

- [x] **Sprint 3.4: Directed (Bayes) vs Undirected (Markov) Comparison (Phase 22)**
  - [x] Formal theoretical comparison matrix across graph types, dependency formulations, spatial scale, and inference goals.

- [x] **Sprint 3.5: Master Integrated Results Matrix (Phase 25)**
  - [x] Produce `outputs/final_retinal_pgm_results.csv` integrating mathematical features, BN posterior probabilities, and MRF metrics.

- [ ] **Sprint 3.6: Resolution of 8 Core Research Experiments (Phase 26)**
  - [ ] Write evidence-based answers once limitations 1–5 above are addressed.

- [x] **Sprint 3.7: Interactive Notebook & Presentation Visualizations (Phases 24, 35, 36)**
  - [x] Demonstration Jupyter Notebook (`notebooks/retinal_pgm_pipeline_demo.ipynb`).
  - [x] `src/visualization.py`: 15 figures in `outputs/figures/`, all rendered from pipeline outputs (see `outputs/figures/README.md`).
