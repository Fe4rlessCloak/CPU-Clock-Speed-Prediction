# CPU Clock Speed Prediction

Predicting Intel CPU maximum turbo frequencies using linear regression with engineered semiconductor features.

## Data

The dataset is sourced from [intel-processors](https://github.com/toUpperCase78/intel-processors) by toUpperCase78. It includes specifications across multiple Intel processor families (Atom, Celeron, Core, Core Ultra, Pentium, Xeon, and others).

## Approach

Linear regression is used to model the relationship between processor features and clock speed targets. The model iterates through seven phases, progressively introducing categorical context, physical constraints, and engineered features while preserving the linear framework required for extrapolation to future architectures.

## Structure

```
.
├── data/           # CSV datasets by processor family
├── model/          # Model training and evaluation
└── README.md
```

## Setup

```bash
conda create -n data python=3.12 pandas scikit-learn
conda activate data
```

## Usage

```bash
python model/main.py
```

## Model Evolution

Predicting a CPU's maximum turbo frequency using linear regression is a challenge because silicon scaling behavior is step-conditional and heavily segmented by marketing tiers. This section outlines how the model was iteratively refined from a blunt baseline into a physically grounded extrapolation engine.

### Phase 1: The Baseline

- **Strategy:** Raw, unmapped continuous specs (`Lithography(nm)`, `Cores`, `TDP(W)`) fed into vanilla Linear Regression.
- **Metrics:** Training R-squared ~38% | Test RMSE ~0.65 GHz
- **Insight:** The model was unbiased (normally distributed residuals) but highly imprecise -- missing targets by an average of 650 MHz because it was blind to architectural differences.

### Phase 2: Segmentation and Temporal Context

- **Strategy:** Introduced `Release Date` as a temporal feature and one-hot encoded vertical product segments (Xeon, Core, Core Ultra, Pentium, Celeron).
- **Metrics:** Test R-squared ~62.3% | Test MAE 0.385 GHz
- **Insight:** A massive leap forward. By providing categorical market boundaries, the model successfully mapped distinct frequency profiles across different tiers, nearly doubling predictive accuracy.

### Phase 3: Suffix Tiering and the Brute-Force Approach

- **Strategy:** Extracted CPU SKU suffixes into performance classes (High Perf, Power Optimized, Ultra-Low Power). Tested a full feature space with 27 dummy variables.
- **Metrics:** Test R-squared 75.68% | Test MAE 0.303 GHz
- **Insight:** The inflated score was a double-edged illusion. The feature space suffered from multicollinearity (Cores vs Threads correlation of 0.98), and the model cheated by including `Base Freq.(GHz)` -- using the base clock as a deterministic shortcut. Stripping this shortcut dropped the realistic baseline to ~72%.

### Phase 4: Physics Injection

- **Strategy:** Compressed 91 marketing codenames into 26 microarchitectural families. Injected physical constraints: `L2_per_P_Core_KB`, `Is_Tiled`, `Is_Mesh`, and `Log_Node_Density`. Applied Ridge Regression to penalize unstable coefficients.
- **Metrics:** Test R-squared 66.39% | Test MAE 0.378 GHz
- **Insight:** Coefficients aligned with real-world semiconductor physics -- the model correctly penalized mesh topologies for routing overhead and high node densities for thermal limits. The 66% ceiling confirmed that a single global linear line has structural limitations across diverse hardware eras.

### Phase 5: Random Forest

- **Strategy:** Swapped the linear engine for `RandomForestRegressor` to test non-linear handling of the engineered features.
- **Metrics:** Training R-squared 95.80% | Test R-squared 85.32% | Test MAE 0.211 GHz
- **Insight:** Tree-based ensembles cannot extrapolate outside training data boundaries. Since a project goal is to project trends into future architectures, a model that flatlines at historical endpoints is non-viable.

### Phase 6: The 70% Ceiling

- **Strategy:** Returned to Ridge Regression to preserve forward-looking extrapolation. Retained engineered physical features, introduced temporal node maturity tracking (`Node_Maturity_Years`), and tuned the OLS pipeline.
- **Metrics:** Training R-squared 0.7032 | Testing R-squared 0.6851 | Test MAE 0.3680 GHz
- **Insight:** The linear model settled near 70% variance captured -- a hard structural ceiling. Consumer silicon arcs upward with core counts while server silicon slopes downward due to die size and mesh fabric drag. A single global regression line, forced to average across both domains, could push no further.

### Phase 7: Breaking the Plateau

- **Strategy:** Three simultaneous structural changes. First, the raw `Threads` column was converted into `Threads_per_Core` -- an SMT ratio that eliminated the 0.98 Cores-Threads correlation while preserving the exact thermal penalty that Hyper-Threading imposes on single-core boost clocks. Second, two per-core power metrics were engineered: `TDP_per_Core` and the `Power_Starvation_Index` (Cores^2 / TDP), giving the purely additive linear model the geometric coordinates to recognize that a 125W 8-core chip allocates far more headroom per core than a 350W 56-core server slab. Third, the entire enterprise server segment (Xeon Scalable and Xeon Legacy) was surgically removed from the training set, optimizing the pipeline exclusively for consumer desktop and mobile silicon.

![Consumer silicon arcs upward with core count while Xeon server silicon slopes downward -- the two domains obey opposing physical trends, making a single compromise regression line structurally inadequate.](img/Why-We-Removed-Xeons.png)

- **Metrics:** Test R-squared exceeding 84% | Generalization gap narrowed substantially
- **Verdict:** This was the definitive breakthrough. Linear regression is a conditional expectation engine -- it always seeks the optimal average. By removing the Xeon domain, the model was freed from forcing a mediocre compromise line between two opposing physical realities. The per-core ratio features gave the straight line the curvature it structurally lacks, and the SMT ratio resolved the multicollinearity that had silently destabilized coefficients since Phase 3. The result is a purely linear, consumer-optimized extrapolation engine with genuine predictive range into future architectures.

## Feature Set

**Target:** `Max. Turbo Freq.(GHz)`

### Raw Specifications

| Feature | Description |
|---|---|
| `Lithography(nm)` | Manufacturing process node size |
| `Cores` | Physical core count |
| `TDP(W)` | Thermal Design Power |
| `Release Date` | Year of launch (extracted from release string) |

Raw `Threads` is collected at input but converted to the ratio `Threads_per_Core` before entering the model, resolving the 0.98 Cores-Threads correlation.

### Engineered Physical Features

| Feature | Description |
|---|---|
| `L2_per_Core_KB` | L2 cache per P-core, mapped from microarchitecture lookup tables |
| `Is_Tiled` | Binary flag for MCM / Foveros tiled packaging |
| `Is_Mesh` | Binary flag for server mesh interconnect topology |
| `Log_Node_Density` | Log-transformed estimated transistor density (MTr/mm2) |
| `Node_Maturity_Years` | Years elapsed since the process node was first introduced |

### Derived Per-Core Ratios

| Feature | Description |
|---|---|
| `TDP_per_Core` | TDP divided by Cores -- localized thermal headroom per core |
| `Threads_per_Core` | Threads divided by Cores -- SMT ratio capturing Hyper-Threading overhead |
| `Power_Starvation_Index` | Cores squared divided by TDP -- penalizes high core counts on thin power envelopes |

### Interaction Features

| Feature | Description |
|---|---|
| `Cores_x_Is_Mesh` | Cores multiplied by mesh flag to capture mesh routing overhead |
| `TDP_x_Is_Tiled` | TDP multiplied by tiled packaging flag |

### One-Hot Encoded Groups

| Group | Levels | Dummies |
|---|---|---|
| `Vertical Segment` | Atom, Celeron, Core, Core Ultra, Intel, Pentium (6 levels; Xeon excluded from training) | 5 |
| `Power Tier` | Embedded, Extreme Low Power, High Perf, High Perf Mobile, Low Power, Mobile (Legacy), No Graphics, Power Optimized, Standard, Standard / Graphics, Ultra-Low Power, BGA / Soldered (12 levels) | 11 |

**Total: 30 columns** (14 numeric + 16 dummy). All numeric features are standardized via `StandardScaler` before training.

## License

Data attributed to [toUpperCase78/intel-processors](https://github.com/toUpperCase78/intel-processors). Code under MIT.
