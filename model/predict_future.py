"""
Predict turbo frequency for a future / hypothetical consumer CPU.

Usage:
    python predict_future.py          (not conda run)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path(".")

# ── 1. Load saved artifacts ──────────────────────────────────────
model = joblib.load(MODEL_DIR / "saved_model.joblib")
scaler = joblib.load(MODEL_DIR / "saved_scaler.joblib")
all_columns = joblib.load(MODEL_DIR / "saved_columns.joblib")
family_dummy_cols = joblib.load(MODEL_DIR / "saved_family_dummies.joblib")
tier_dummy_cols = joblib.load(MODEL_DIR / "saved_tier_dummies.joblib")

print("=" * 60)
print("  CPU Clock Speed Predictor (Ridge, Consumer-Only)")
print("=" * 60)

# ── 2. Collect inputs ────────────────────────────────────────────

print("\n--- Raw Specs ---")
lithography = float(input("  Lithography(nm)       [e.g. 3]: "))
cores = float(input("  Cores                 [e.g. 8]: "))
threads = float(input("  Threads               [e.g. 16]: "))
tdp = float(input("  TDP(W)                [e.g. 125]: "))
release_year = float(input("  Release Year          [e.g. 2026]: "))

print("\n--- Engineered Physical Features ---")
print("  (Hint: Alder Lake=1280, Raptor Lake=2048, Arrow Lake=3072, etc.)")
l2_per_core = float(input("  L2 per P-core (KB)    [e.g. 3072]: "))
is_tiled = int(input("  Is Tiled? (0/1)       [e.g. 1]: "))
is_mesh = int(input("  Is Mesh?  (0/1)       [e.g. 0]: "))
node_density = float(input("  Node Density (MTr/mm2) [e.g. 200]: "))
node_maturity = float(input("  Node Maturity (years)  [e.g. 2]: "))

print("\n--- One-Hot: Vertical Segment ---")
print("  Options: Atom, Celeron, Core, Core Ultra, Intel, Pentium")
family = input("  Family: ").strip()

print("\n--- One-Hot: Power Tier ---")
print("  Options: Standard, High Perf, High Perf Mobile, Ultra-Low Power,")
print("           Extreme Low Power, Power Optimized, Low Power, Embedded,")
print("           No Graphics, Mobile (Legacy), Standard / Graphics,")
print("           BGA / Soldered")
tier = input("  Power Tier: ").strip()

# ── 3. Build feature vector ─────────────────────────────────────

log_node_density = np.log1p(node_density)
cores_x_mesh = cores * is_mesh
tdp_x_tiled = tdp * is_tiled
tdp_per_core = tdp / cores
threads_per_core = threads / cores
power_starvation = (cores ** 2) / tdp

feature_dict = {
    "Lithography(nm)": lithography,
    "Cores": cores,
    "TDP(W)": tdp,
    "Release Date": release_year,
    "L2_per_Core_KB": l2_per_core,
    "Is_Tiled": is_tiled,
    "Is_Mesh": is_mesh,
    "Log_Node_Density": log_node_density,
    "Cores_x_Is_Mesh": cores_x_mesh,
    "TDP_x_Is_Tiled": tdp_x_tiled,
    "Node_Maturity_Years": node_maturity,
    "TDP_per_Core": tdp_per_core,
    "Threads_per_Core": threads_per_core,
    "Power_Starvation_Index": power_starvation,
}

# Add one-hot family dummies
family_prefix = "fam_"
for col in family_dummy_cols:
    dummy_family = col.replace(family_prefix, "").replace("_", " ").strip()
    feature_dict[col] = 1 if family.lower() == dummy_family.lower() else 0

# Add one-hot tier dummies
tier_prefix = "tier_"
for col in tier_dummy_cols:
    dummy_tier = col.replace(tier_prefix, "").replace("_", " ").strip()
    feature_dict[col] = 1 if tier.lower() == dummy_tier.lower() else 0

# Build DataFrame in the exact column order the model expects
input_df = pd.DataFrame([feature_dict])[all_columns]

# ── 4. Scale & predict ───────────────────────────────────────────
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# ── 5. Show result ───────────────────────────────────────────────
print()
print("=" * 60)
print("  PREDICTION")
print("=" * 60)
print(f"  Estimated Max. Turbo Freq.:  {prediction:.3f} GHz")
print(f"                              ~ {prediction*1000:.0f} MHz")
print()

# Show what the model used
print("--- Input Summary ---")
print(f"  {lithography} nm | {cores:.0f}C/{threads:.0f}T | {tdp} W | {int(release_year)}")
print(f"  L2: {l2_per_core} KB/core | Tiled: {bool(is_tiled)} | Mesh: {bool(is_mesh)}")
print(f"  Density: {node_density} MTr/mm2 | Maturity: {node_maturity} yrs")
print(f"  TDP/Core: {tdp_per_core:.1f} W | Thr/Core: {threads_per_core:.2f}")
print(f"  Power Starvation: {power_starvation:.2f}")
print(f"  Family: {family}  |  Tier: {tier}")
