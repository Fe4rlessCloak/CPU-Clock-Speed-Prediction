"""
Train the full Ridge model on CONSUMER historical data and save weights + scaler.
Xeon server processors are excluded to avoid forcing a compromise between
consumer and server silicon physics.

Run once, then use predict_future.py for manual predictions.
"""

import pandas as pd
import numpy as np
import re
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── 1. Configuration ──────────────────────────────────────────────
DATA_DIR = Path("../data")
MODEL_DIR = Path(".")
MODEL_PATH = MODEL_DIR / "saved_model.joblib"
SCALER_PATH = MODEL_DIR / "saved_scaler.joblib"
COLS_PATH = MODEL_DIR / "saved_columns.joblib"
FAMILY_DUMMIES_PATH = MODEL_DIR / "saved_family_dummies.joblib"
TIER_DUMMIES_PATH = MODEL_DIR / "saved_tier_dummies.joblib"

# ── 2. Lookup tables (from notebook) ──────────────────────────────

def classify_suffix(name):
    m = re.search(r'(\d+)([A-Z]+[0-9]?)$', name)
    suffix = m.group(2) if m else ""
    if suffix in ("K", "KF", "KS", "X", "XE", "HK", "HX"):
        return "High Perf"
    elif suffix in ("H", "HQ", "HL", "HE", "QM", "MQ"):
        return "High Perf Mobile"
    elif suffix in ("U", "UM", "UL", "UE"):
        return "Ultra-Low Power"
    elif suffix == "Y":
        return "Extreme Low Power"
    elif suffix in ("T", "TE"):
        return "Power Optimized"
    elif suffix in ("N", "NT", "NTE"):
        return "Low Power"
    elif suffix in ("E", "EQ", "EC"):
        return "Embedded"
    elif suffix in ("F", "KF"):
        return "No Graphics"
    elif suffix in ("M",):
        return "Mobile (Legacy)"
    elif suffix in ("S", "P", "G", "G1", "G4", "G7"):
        return "Standard / Graphics"
    elif suffix in ("B", "R", "RE", "RK"):
        return "BGA / Soldered"
    else:
        return "Standard"

gen_map = {
    "Merom": "Core 2", "Conroe": "Core 2", "Kentsfield": "Core 2",
    "Wolfdale": "Core 2", "Yorkfield": "Core 2", "Penryn": "Core 2",
    "Bloomfield": "Nehalem", "Lynnfield": "Nehalem", "Nehalem EP": "Nehalem",
    "Nehalem EX": "Nehalem", "Gulftown": "Westmere", "Clarkdale": "Westmere",
    "Arrandale": "Westmere", "Westmere EP": "Westmere", "Westmere EX": "Westmere",
    "Sandy Bridge": "Sandy Bridge", "Sandy Bridge E": "Sandy Bridge",
    "Sandy Bridge EN": "Sandy Bridge", "Sandy Bridge EP": "Sandy Bridge",
    "Ivy Bridge": "Ivy Bridge", "Ivy Bridge E": "Ivy Bridge",
    "Ivy Bridge EN": "Ivy Bridge", "Ivy Bridge EP": "Ivy Bridge",
    "Haswell": "Haswell", "Haswell E": "Haswell",
    "Broadwell": "Broadwell", "Broadwell E": "Broadwell",
    "Skylake": "Skylake", "Kaby Lake": "Kaby Lake",
    "Kaby Lake G": "Kaby Lake", "Kaby Lake R": "Kaby Lake",
    "Coffee Lake": "Coffee Lake", "Comet Lake": "Comet Lake",
    "Whiskey Lake": "Whiskey Lake", "Amber Lake": "Amber Lake",
    "Amber Lake Y": "Amber Lake", "Ice Lake": "Ice Lake",
    "Tiger Lake": "Tiger Lake", "Rocket Lake": "Rocket Lake",
    "Alder Lake": "Alder Lake", "Alder Lake-N": "Alder Lake",
    "Raptor Lake": "Raptor Lake", "Meteor Lake": "Meteor Lake",
    "Lunar Lake": "Lunar Lake", "Arrow Lake": "Arrow Lake",
    "Panther Lake": "Panther Lake",
    "Sapphire Rapids": "Xeon Scalable", "Sapphire Rapids HBM": "Xeon Scalable",
    "Sapphire Rapids Edge Enhanced": "Xeon Scalable",
    "Emerald Rapids": "Xeon Scalable", "Cascade Lake": "Xeon Scalable",
    "Cooper Lake": "Xeon Scalable", "Denverton": "Atom",
    "Rangeley": "Atom", "Avoton": "Atom", "Centerton": "Atom",
    "Braswell": "Atom", "Cherry Trail": "Atom",
    "Bay Trail": "Atom", "Merrifield": "Atom",
    "Moorefield": "Atom", "Jasper Lake": "Atom",
    "Elkhart Lake": "Atom", "Amston Lake": "Atom",
    "Cloverview": "Atom", "Cedarview": "Atom",
    "Penwell": "Atom", "SoFIA 3G R": "Atom", "SoFIA LTE": "Atom",
    "Twin Lake": "Atom", "Gemini Lake": "Atom",
    "Gemini Lake Refresh": "Atom", "Snow Ridge": "Atom",
    "Parker Ridge": "Atom", "Hewitt Lake": "Atom",
    "Lakefield": "Hybrid", "Jasper Forest": "Xeon Legacy",
    "Nocona": "Xeon Legacy", "Prestonia": "Xeon Legacy",
    "Sossaman": "Xeon Legacy", "Tigerton": "Xeon Legacy",
    "Dunnington": "Xeon Legacy", "Clovertown": "Xeon Legacy",
    "Harpertown": "Xeon Legacy", "Woodcrest": "Xeon Legacy",
    "Gladden": "Xeon Legacy", "Crystal Well": "Haswell",
    "Devil's Canyon": "Haswell",
}

arch_features = {
    "Core 2":        [3072, 0, 0, 4.0],
    "Nehalem":       [256,  0, 0, 4.0],
    "Westmere":      [256,  0, 0, 9.0],
    "Sandy Bridge":  [256,  0, 0, 9.0],
    "Ivy Bridge":    [256,  0, 0, 17.0],
    "Haswell":       [256,  0, 0, 17.0],
    "Broadwell":     [256,  0, 0, 37.5],
    "Skylake":       [256,  0, 0, 37.5],
    "Kaby Lake":     [256,  0, 0, 37.5],
    "Coffee Lake":   [256,  0, 0, 37.5],
    "Comet Lake":    [256,  0, 0, 37.5],
    "Whiskey Lake":  [256,  0, 0, 37.5],
    "Amber Lake":    [256,  0, 0, 37.5],
    "Ice Lake":      [512,  0, 0, 63.0],
    "Tiger Lake":    [1280, 0, 0, 63.0],
    "Rocket Lake":   [512,  0, 0, 37.5],
    "Alder Lake":    [1280, 0, 0, 100.0],
    "Raptor Lake":   [2048, 0, 0, 100.0],
    "Meteor Lake":   [2048, 1, 0, 100.0],
    "Lunar Lake":    [2560, 1, 0, 170.0],
    "Arrow Lake":    [3072, 1, 0, 170.0],
    "Panther Lake":  [3072, 1, 0, 150.0],
    "Xeon Scalable": [2048, 1, 1, 100.0],
    "Atom":          [512,  0, 0, 100.0],
    "Hybrid":        [512,  1, 0, 63.0],
    "Xeon Legacy":   [256,  0, 0, 17.0],
}

node_intro_map = {
    "Core 2": 2007, "Nehalem": 2007,
    "Westmere": 2010, "Sandy Bridge": 2010,
    "Ivy Bridge": 2012, "Haswell": 2012,
    "Broadwell": 2014, "Skylake": 2014, "Kaby Lake": 2014,
    "Coffee Lake": 2014, "Comet Lake": 2014, "Whiskey Lake": 2014,
    "Amber Lake": 2014, "Rocket Lake": 2014,
    "Ice Lake": 2019, "Tiger Lake": 2019, "Alder Lake": 2019,
    "Raptor Lake": 2019, "Atom": 2019, "Hybrid": 2019,
    "Meteor Lake": 2023,
    "Lunar Lake": 2024, "Arrow Lake": 2024,
    "Panther Lake": 2025,
    "Xeon Scalable": 2017,
    "Xeon Legacy": 2012,
}

# ── 3. Load data (CONSUMER ONLY -- exclude Xeon) ───────────────────

csv_files = [
    DATA_DIR / "Core-Processors-1.8.csv",
    DATA_DIR / "Atom-Processors-1.9-16col.csv",
    DATA_DIR / "Celeron-Processors-1.9-16col.csv",
    DATA_DIR / "Core_Ultra-Processors-1.10-16col.csv",
    DATA_DIR / "Intel-Processors-1.9-16col.csv",
    DATA_DIR / "Pentium-Processors-1.9-16col.csv",
]

frames = []
for f in csv_files:
    df = pd.read_csv(f)
    family = f.stem.split("-")[0]
    df["Vertical Segment"] = family
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

family_map = {
    "Atom": "Atom", "Celeron": "Celeron", "Core": "Core",
    "CoreUltra": "Core Ultra", "Intel": "Intel",
    "Pentium": "Pentium",
}

df["Release Date"] = df["Release Date"].str[-2:].astype(int) + 2000
df["Power Tier"] = df["Product"].apply(classify_suffix)
df["Arch Group"] = df["Code Name"].map(gen_map)

# ── 4. Engineered features ────────────────────────────────────────

for group, vals in arch_features.items():
    mask = df["Arch Group"] == group
    df.loc[mask, "L2_per_Core_KB"] = vals[0]
    df.loc[mask, "Is_Tiled"] = vals[1]
    df.loc[mask, "Is_Mesh"] = vals[2]
    df.loc[mask, "Node_Density"] = vals[3]

numeric_cols = ["Lithography(nm)", "Cores", "TDP(W)", "Threads",
                "L2_per_Core_KB", "Is_Tiled", "Is_Mesh", "Node_Density"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Log_Node_Density"] = np.log1p(df["Node_Density"])
df["Cores_x_Is_Mesh"] = df["Cores"] * df["Is_Mesh"]
df["TDP_x_Is_Tiled"] = df["TDP(W)"] * df["Is_Tiled"]

# New features: node maturity, per-core ratios
df["Node_Intro_Year"] = df["Arch Group"].map(node_intro_map)
df["Node_Maturity_Years"] = df["Release Date"] - df["Node_Intro_Year"]
df["TDP_per_Core"] = df["TDP(W)"] / df["Cores"]
df["Threads_per_Core"] = df["Threads"] / df["Cores"]
df["Power_Starvation_Index"] = (df["Cores"] ** 2) / df["TDP(W)"]
df.drop(columns=["Node_Density", "Node_Intro_Year"], inplace=True)

y = pd.to_numeric(df["Max. Turbo Freq.(GHz)"], errors="coerce")

# ── 5. One-hot encode ────────────────────────────────────────────

family_dummies = pd.get_dummies(df["Vertical Segment"], prefix="fam", drop_first=True)
tier_dummies = pd.get_dummies(df["Power Tier"], prefix="tier", drop_first=True)

base_features = [
    "Lithography(nm)", "Cores", "TDP(W)", "Release Date",
    "L2_per_Core_KB", "Is_Tiled", "Is_Mesh", "Log_Node_Density",
    "Cores_x_Is_Mesh", "TDP_x_Is_Tiled",
    "Node_Maturity_Years", "TDP_per_Core", "Threads_per_Core",
    "Power_Starvation_Index",
]

X = pd.concat([df[base_features], family_dummies, tier_dummies], axis=1)

# ── 6. Drop NaN rows ─────────────────────────────────────────────

mask = X.isna().any(axis=1) | y.isna()
X = X[~mask]
y = y[~mask]

print(f"Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Feature columns: {list(X.columns)}")

# ── 7. Scale & train ─────────────────────────────────────────────

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = Ridge()
model.fit(X_train, y_train)

# ── 8. Evaluate ──────────────────────────────────────────────────

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\nTraining set: R-squared = {r2_score(y_train, y_pred_train):.4f}, "
      f"MAE = {mean_absolute_error(y_train, y_pred_train):.4f} GHz")
print(f"Test set:     R-squared = {r2_score(y_test, y_pred_test):.4f}, "
      f"MAE = {mean_absolute_error(y_test, y_pred_test):.4f} GHz")

# ── 9. Save everything ───────────────────────────────────────────

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(X.columns.tolist(), COLS_PATH)
joblib.dump(family_dummies.columns.tolist(), FAMILY_DUMMIES_PATH)
joblib.dump(tier_dummies.columns.tolist(), TIER_DUMMIES_PATH)

print(f"\nSaved to:")
print(f"  {MODEL_PATH}")
print(f"  {SCALER_PATH}")
print(f"  {COLS_PATH}")
print(f"  {FAMILY_DUMMIES_PATH}")
print(f"  {TIER_DUMMIES_PATH}")
