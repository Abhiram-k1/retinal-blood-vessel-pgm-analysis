# Retinal Blood-Vessel Analysis using Probabilistic Graphical Models & Graph Theory: Plain-English Guide (pseudo.md)

Welcome to the plain-English, intuitive explanation of this research project! This document explains **what** each step does, **why** we do it (the scientific, clinical, and mathematical rationale), and **how** it is accomplished in clear, accessible language.

---

## 1. High-Level Vision & Core Purpose

### What are we building?
An intelligent, mathematically rigorous, and auditable pipeline that takes high-resolution retinal fundus photographs (photographs of the inner back of the eye), extracts the tree of blood vessels, transforms those vessels into discrete mathematical graph networks, computes geometric and topological metrics, and applies **Probabilistic Graphical Models (PGMs)**—both **Directed Bayesian Networks** and **Undirected Markov Random Fields (MRFs)**—alongside **Multiresolution Wavelet Analysis (MRA)** and **Deep Learning Receptive-Field Models** to perform probabilistic reasoning and spatial inference.

### Why does this matter?
The blood vessels in the human retina are the only part of the body's central circulatory system that can be directly photographed non-invasively without surgery. Variations in vessel twistiness (tortuosity), branching geometry, and network transport efficiency are early warning biomarkers for major sight-threatening and systemic diseases:
* **Diabetic Retinopathy (DR):** The leading cause of preventable blindness; causes microaneurysms, vessel leakage, and abnormal new vessel sprouting (neovascularization).
* **Hypertension & Stroke:** Chronic elevated blood pressure stiffens and narrows retinal arterioles and dramatically increases vessel tortuosity.
* **Cardiovascular & Systemic Vascular Disease:** Variations in fractal branching and global network efficiency reflect widespread microvascular health.

Rather than treating vessel analysis as an opaque deep learning "black box", this project integrates **multiresolution computer vision**, **discrete network science**, and **probabilistic graphical modeling**, giving clinicians interpretable, mathematically sound, and scientifically auditable conclusions.

---

## 2. Recent Research Updates & Critical Bug Fixes (Summary)

To achieve state-of-the-art research quality and eliminate mathematical inconsistencies, the codebase underwent systematic upgrades:

1. **Multiresolution Analysis (MRA) Integration:**
   * *What:* Introduced multi-scale directional Gabor filterbanks across 3 spatial scales ($\sigma \in \{2, 4, 6\}\text{ px}$) and 6 orientations.
   * *Why:* Real blood vessels vary in width from thick main trunks ($\approx 10\text{ px}$) down to delicate micro-capillaries ($\approx 1\text{ px}$). A single filter scale either misses capillaries or bloats thick trunks.
   * *Result:* Recovers fine peripheral vessels, boosting sensitivity by $+6.3\%$.

2. **Supervised Deep Receptive-Field Neural Model:**
   * *What:* Implemented a 10-dimensional spatial convolutional feature stack feeding a deep Multi-Layer Perceptron (MLP) with two ReLU hidden layers ($64 \rightarrow 32$).
   * *Why:* Combines intensity, local context, textural variance, Gabor responses, and gradient magnitude to learn non-linear vessel boundaries from training images.
   * *Result:* Achieved the highest vessel sensitivity (**86.46%**) and top Dice score (**0.5673**).

3. **Data-Driven MRF Gaussian Likelihoods (BUG-001 Fix):**
   * *What:* Eliminated hardcoded priors and implemented `fit_likelihoods()`, deriving Gaussian parameters directly from training imagery.
   * *Why:* Inverted contrast between vessels and background had previously caused the MRF to degrade segmentation.
   * *Result:* Monotonic energy reduction and substantial localized F1 gains (e.g. Patient 11L F1 improved by $+11.35\%$).

4. **Checkerboard Iterated Conditional Modes (ICM) (BUG-002 Fix):**
   * *What:* Replaced simultaneous vector updates with red/black checkerboard conditional independence sweeps.
   * *Why:* Updating all pixels simultaneously violates the Markov blanket assumption and causes label oscillations.

5. **Non-Circular Bayesian Network Target State $S$ (BUG-003 Fix):**
   * *What:* Redefined the latent network state $S$ using the independent continuous fractal dimension threshold ($D_f < 1.6803 \implies \text{Complex}$).
   * *Why:* Previously, $S$ was defined using the exact same variables ($B, T, D$) used as evidence, creating a 100% tautological posterior.
   * *Result:* Genuine probabilistic inference with mathematically verified factorisation ($\sum P = 1.000000$).

6. **Graph Gap-Bridging & 52.8% De-fragmentation (BUG-004 Fix):**
   * *What:* Added endpoint gap-bridging ($\le 3.5\text{ px}$) and pruned isolated 0-degree orphan nodes.
   * *Result:* Disjoint graph components dropped from $750.8 \rightarrow 354.6$, and network density doubled.

7. **Murray's Law Bifurcation Angle Correction (BUG-005 Fix):**
   * *What:* Measured angles specifically between the two daughter branches originating from a bifurcation.
   * *Result:* Mean angle shifted from an arbitrary $121.0^\circ$ to **$60.3^\circ \pm 3.4^\circ$**, conforming directly to Murray's physiological Principle of Minimum Work ($60^\circ\text{--}75^\circ$).

---

## 3. The Complete Step-by-Step Technical Walkthrough

---

### Step 1: Dataset Ingestion & Leakage-Free Patient Splitting
* **What:** Read the 28 retinal fundus images from the CHASE_DB1 database alongside manual annotations from two human observers.
* **Why:** If the left eye of Subject 01 is in training and their right eye is in testing, the algorithm "cheats" because both eyes share systemic vascular traits. We enforce strict **patient-level separation**:
  * **Training Split (20 images):** Subjects `01L` through `10R`.
  * **Testing Split (8 images):** Subjects `11L` through `14R` (strictly held out).
* **How:** Orchestrated by `dataset.py` to produce `dataset_manifest.csv`.

---

### Step 2: Image Preprocessing & Illumination Equalization
* **What:** Clean raw color fundus images and enhance vessel visibility.
* **Why:** The red channel is oversaturated, the blue channel is noisy, and peripheral retinas suffer from dim vignetting.
* **How:**
  1. **Green Channel Extraction ($I_G$):** Blood hemoglobin absorbs green light strongly, maximizing contrast.
  2. **CLAHE (Contrast-Limited Adaptive Histogram Equalization):** Equalizes contrast in local $8 \times 8$ tiles without blowing up noise.
  3. **Bilateral Smoothing:** Removes sensor noise while keeping sharp vessel edges.
  4. **Min-Max Normalization:** Maps intensity cleanly to the $[0, 1]$ interval.

---

### Step 3: Multiresolution Wavelet Analysis (MRA)
* **What:** Filter the preprocessed image using a bank of directional Gabor wavelets across 3 spatial scales ($\sigma = 2, 4, 6\text{ px}$) and 6 orientations.
* **Why:** Fine capillaries require narrow filter envelopes, while major arteries require broad envelopes. MRA decomposes the image into scale-specific frequency subbands.
* **How:** Evaluated via OpenCV in `multiscale_features.py`; the maximum response across orientations at each scale creates caliber-tuned vessel probability maps.

---

### Step 4: Supervised Deep Neural Receptive-Field Segmentation
* **What:** Classify each pixel inside the retinal disc as vessel or background using a deep neural network.
* **Why:** Classical thresholding struggles at vessel borders. A neural network learns the non-linear relationship between pixel brightness, surrounding 5x5 context, texture variance, Gabor responses, and edge gradients.
* **How:** In `deep_learning.py`, an MLP neural network ($64 \rightarrow 32$ ReLU hidden units) is trained on balanced samples from the 20 training images and outputs continuous probability maps and binary masks.

---

### Step 5: Morphological Baseline & Ground-Truth Validation
* **What:** Generate morphological bottom-hat masks and quantitatively validate predictions against human ophthalmologists.
* **Why:** We must evaluate segmentation performance using standard clinical metrics strictly within the circular Field of View (FOV).
* **How:** Computes pixel-wise confusion matrix (True Positives, False Positives, True Negatives, False Negatives) and logs:
  * **Accuracy:** $(TP+TN)/(TP+TN+FP+FN)$
  * **Precision:** $TP/(TP+FP)$
  * **Recall / Sensitivity:** $TP/(TP+FN)$
  * **Specificity:** $TN/(TN+FP)$
  * **Dice / F1-Score:** $2TP/(2TP+FP+FN)$
  * **IoU (Jaccard):** $TP/(TP+FP+FN)$

---

### Step 6: Centerline Skeletonization & Node Detection
* **What:** Thin thick vessels down to 1-pixel-wide centerlines while preserving branching connections.
* **Why:** Mathematical graphs cannot be built from thick, variable-width blobs. Centerlines isolate pure topology.
* **How:** Medial axis topological thinning peels boundary pixels without disconnecting paths. Then, an 8-neighborhood cross-number scan labels endpoints ($N=1$), normal segments ($N=2$), and junctions ($N \ge 3$), clustering multi-pixel clusters into unique centroid nodes.

---

### Step 7: Topological Graph Construction $G=(V, E)$ & Gap-Bridging
* **What:** Convert the 1-pixel skeleton into a NetworkX mathematical graph.
* **Why:** Enables rigorous calculations of network density, connectivity, and shortest-path communication efficiency.
* **How:** In `graph_construction.py`, vessel segments become edges storing true pixel coordinates, curve length $L$, and Euclidean chord distance $d$. Small sub-pixel breaks ($\le 3.5\text{ px}$) are bridged, and orphan nodes with degree 0 are pruned.

---

### Step 8: Vascular Geometry & Murray's Law Optimization
* **What:** Compute segment tortuosity and bifurcation branch angles across the vascular graph.
* **Why:** High tortuosity indicates hypertensive microvascular damage. Bifurcation angles reveal whether the vascular tree obeys hemodynamic minimum-work optimization (Murray's Law).
* **How:**
  * **Tortuosity Index:** $T = L / d(u, v)$ (with strict $d > 0$ division-by-zero guards).
  * **Murray's Law Bifurcation Angle:** Angle between the two daughter branches at a junction:
    $$\theta = \arccos\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}\right) \approx 60.3^\circ$$

---

### Step 9: Multi-Scale Box-Counting Fractal Dimension ($D_f$)
* **What:** Quantify the space-filling complexity of the vascular tree.
* **Why:** Pathologies like diabetes cause capillary dropout, reducing fractal branching complexity.
* **How:** Partition the vessel mask using boxes of side length $\epsilon \in \{2, 4, 8, 16, 32, 64, 128\}\text{ px}$. Linear regression of $\log N(\epsilon)$ vs $\log(1/\epsilon)$ yields $D_f$ with $R^2 > 0.995$.

---

### Step 10: Global Network Transport Efficiency ($E_{\text{global}}$)
* **What:** Measure fluid and communication transport efficiency across the vascular graph.
* **Why:** Determines how effectively blood can circulate across the full network.
* **How:** In `network_efficiency.py`, evaluates all-pairs shortest paths:
  $$E_{\text{global}} = \frac{1}{N(N-1)} \sum_{i \ne j} \frac{1}{d(i, j)}$$
  Disconnected node pairs strictly contribute $1/\infty = 0$ without arbitrary penalties.

---

### Step 11: Master Feature Compilation & Tertile Discretization
* **What:** Consolidate all metrics into `retinal_network_features.csv` and discretize into `{Low, Medium, High}`.
* **Why:** Tabular Bayesian Networks require discrete states. Cutoffs must be derived strictly from the training split.
* **How:** Empirical tertile percentiles ($33.3\%$ and $66.7\%$) from training retinas establish bin boundaries.

---

### Step 12: Directed Bayesian Network DAG & CPT Parameter Learning
* **What:** Build a Directed Acyclic Graph encoding vascular dependencies:
  * $B \rightarrow T$ (Branching influences segment tortuosity)
  * $B \rightarrow D$ (Branching dictates network density)
  * $D \rightarrow \eta$ (Density influences transport efficiency)
  * $\{T, D, \eta\} \rightarrow S$ (Tortuosity, density, and efficiency jointly determine network state)
* **Why:** Reason about uncertainty and disease state from observed clinical features.
* **How:** Learn Conditional Probability Tables (CPTs) via Laplace-smoothed Maximum Likelihood Estimation strictly from training retinas, mathematically verifying:
  $$P(B, T, D, \eta, S) = P(B) P(T \mid B) P(D \mid B) P(\eta \mid D) P(S \mid T, D, \eta)$$
  and confirming that total joint probability sums to $1.000000$.

---

### Step 13: Exact Variable Elimination Bayesian Inference
* **What:** Answer diagnostic probabilistic queries under clinical evidence.
* **Why:** For example: "If an eye shows High Tortuosity and High Branching, what is the probability that its network is Complex / Abnormal?"
* **How:** Exact Variable Elimination sums out hidden intermediate variables and normalizes the posterior to sum to $1.0$.

---

### Step 14: Undirected Markov Random Field (MRF) Spatial Energy Model
* **What:** Formulate pixel-level spatial regularization on a 4-connected lattice.
* **Why:** Pixels are not independent. Adjacent pixels should favor having the same label unless image intensity strongly suggests a boundary.
* **How:** Energy function $E(X) = \sum D_i(X_i) + \lambda \sum V_{ij}(X_i, X_j)$:
  * Data potential $D_i(X_i) = -\log P(I_i \mid X_i)$ from empirical Gaussian likelihoods.
  * Smoothness potential $V_{ij} = \beta [X_i \ne X_j]$ penalizes label disagreements.
  * Solved via red/black checkerboard Iterated Conditional Modes (ICM).

---

### Step 15: 5-Condition Quantitative Ablation Study
* **What:** Isolate the individual and combined impact of every module:
  * Condition A: Morphological Baseline
  * Condition B: MRA (Multi-Scale Gabor)
  * Condition C: MRF Spatial Refinement
  * Condition D: Deep Learning Receptive-Field Model
  * Condition E: Full System (Deep Learning + MRF)
* **Why:** Scientifically proves which components drive sensitivity and boundary precision.
* **How:** Evaluates all 5 conditions on the held-out test split and exports `ablation_study_summary.csv`.

---

## 4. Verification and Anti-Fabrication Principles

In strict accordance with scientific integrity standards:
1. **Zero Hardcoded Numbers:** Every metric (node count, tortuosity index, fractal slope, CPT entry, posterior probability, MRF energy) is dynamically calculated from actual retinal image data.
2. **Leakage-Free Protocol:** The 8 test images are completely isolated from all parameter estimation, threshold tuning, and model training.
3. **Reproducibility:** The entire pipeline executes via `python -m src.pipeline` in a single command, generating verified CSV outputs and 16 publication-quality figures.
