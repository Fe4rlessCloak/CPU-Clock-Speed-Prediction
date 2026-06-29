# CPU Clock Speed Prediction

Predicting Intel CPU clock speeds (base, turbo, and boost frequencies) using linear regression.

## Data

The dataset is sourced from [intel-processors](https://github.com/toUpperCase78/intel-processors) by toUpperCase78. It includes specifications across multiple Intel processor families (Atom, Celeron, Core, Core Ultra, Pentium, Xeon, and others).

## Approach

Linear regression is used to model the relationship between processor features (cores, threads, lithography, TDP, cache, memory specs) and clock speed targets.

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

## License

Data attributed to [toUpperCase78/intel-processors](https://github.com/toUpperCase78/intel-processors). Code under MIT.

## Model Evolution

Predicting a CPU's maximum turbo frequency using linear regression is a challenge because silicon scaling behavior is step-conditional and heavily segmented by marketing tiers. This section outlines how the model was iteratively refined from a blunt baseline into a physically grounded extrapolation engine.

### Phase 1: The Baseline

- **Strategy:** Raw, unmapped continuous specs (`Lithography(nm)`, `Cores`, `TDP(W)`) fed into vanilla Linear Regression.
- **Metrics:** Training R²: ~38% | Test RMSE: ~0.65 GHz
- **Insight:** The model was unbiased (normally distributed residuals) but highly imprecise -- missing targets by an average of 650 MHz because it was blind to architectural differences.

### Phase 2: Segmentation and Temporal Context

- **Strategy:** Introduced `Release Date` as a temporal feature and one-hot encoded vertical product segments (`Xeon`, `Core`, `Core Ultra`, `Pentium`, `Celeron`).
- **Metrics:** Test R²: ~62.3% | Test MAE: 0.385 GHz
- **Insight:** A massive leap forward. By providing categorical market boundaries, the model successfully mapped distinct frequency profiles across different tiers, nearly doubling predictive accuracy.

### Phase 3: Suffix Tiering and the Brute-Force Approach

- **Strategy:** Extracted CPU SKU suffixes into performance classes (High Perf, Power Optimized, Ultra-Low Power). Tested a full feature space with 27 dummy variables.
- **Metrics:** Test R²: 75.68% | Test MAE: 0.303 GHz
- **Insight:** The inflated score was a double-edged illusion. The feature space suffered from multicollinearity (Cores vs Threads correlation of 0.98), and the model cheated by including `Base Freq.(GHz)` -- using the base clock as a deterministic shortcut. Stripping this shortcut dropped the realistic baseline to ~72%.

### Phase 4: Physics Injection

- **Strategy:** Dropped `Threads` to resolve multicollinearity and removed the base clock shortcut. Compressed 91 marketing codenames into 26 microarchitectural families. Injected physical constraints: `L2_per_P_Core_KB`, `Is_Tiled`, `Is_Mesh`, and `Log_Node_Density`. Applied Ridge Regression to penalize unstable coefficients.
- **Metrics:** Test R²: 66.39% | Test MAE: 0.378 GHz
- **Insight:** Coefficients aligned with real-world semiconductor physics -- the model correctly penalized mesh topologies for routing overhead and high node densities for thermal limits. The 66% ceiling confirmed that a single global linear line has structural limitations across diverse hardware eras.

### Phase 5: Random Forest

- **Strategy:** Swapped the linear engine for `RandomForestRegressor` to test non-linear handling of the engineered features.
- **Metrics:** Training R²: 95.80% | Test R²: 85.32% | Test MAE: 0.211 GHz
- **Insight:** Tree-based ensembles cannot extrapolate outside training data boundaries. Since a project goal is to project trends into future architectures, a model that flatlines at historical endpoints is non-viable.

### Phase 6: Final Linear Polish

- **Strategy:** Returned to a continuous linear framework to preserve forward-looking extrapolation. Retained engineered physical features, stabilized data matrices via correlation analysis, and tuned the OLS pipeline.
- **Final Metrics:** Training R²: 0.7032 | Testing R²: 0.6851 | Test MAE: 0.3680 GHz
- **Verdict:** By optimizing the feature space rather than changing the math, the linear model captures nearly 70% of historical variance with a narrow generalization gap. It stands as a physically grounded, stable extrapolation engine ready for future node transitions.
