"""
Brute-force linear regression using ALL available columns.
One-hot encodes categoricals, keeps all numerics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re

# ── 1. Load with family tagging ──────────────────────────────────
data_dir = Path("../data")
frames = []
for f in sorted(data_dir.glob("*.csv")):
    df = pd.read_csv(f)
    # Extract family from filename
    family = f.stem.split("-")[0]  # "Core-Ultra", "Atom", etc.
    df["Family"] = family
    frames.append(df)

df = pd.concat(frames, ignore_index=True)

# ── 2. Extract features from text columns ────────────────────────
# Release Year
df["Release Year"] = df["Release Date"].str[-2:].astype(int) + 2000

# Suffix / power tier from product name
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
    elif suffix in ("F",):
        return "No Graphics"
    elif suffix in ("M",):
        return "Mobile Legacy"
    elif suffix in ("S", "P", "G", "G1", "G4", "G7"):
        return "Standard Graphics"
    elif suffix in ("B", "R", "RE", "RK"):
        return "BGA Soldered"
    else:
        return "Standard"

df["Power Tier"] = df["Product"].apply(classify_suffix)

# ── 3. Select which columns to use ────────────────────────────────
# Numeric columns (keep as-is)
numeric_cols = [
    "Cores", "Threads", "Lithography(nm)", "TDP(W)", "Cache(MB)",
    "Max Memory Size(GB)", "Release Year"
]

# Clean them up
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Categorical columns (one-hot encode)
categorical_cols = [
    "Status", "Family", "Power Tier"
]
# Note: Code Name has 91 unique values — skip to avoid 90 sparse columns.
# Memory Types and Cache Info are too free-text to encode cleanly.
# Integrated Graphics is also sparse.

# One-hot encode
encoded_parts = []
for col in categorical_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
    encoded_parts.append(dummies)
    print(f"{col}: {df[col].nunique()} unique → {dummies.shape[1]} dummy columns")

# ── 4. Assemble feature matrix ──────────────────────────────────
X = pd.concat([df[numeric_cols]] + encoded_parts, axis=1)
y = pd.to_numeric(df["Max. Turbo Freq.(GHz)"], errors="coerce")

# Drop rows with any NaN
mask = X.isna().any(axis=1) | y.isna()
X = X[~mask]
y = y[~mask]

print(f"\nTotal features: {X.shape[1]}")
print(f"Clean samples:  {X.shape[0]}")
print()

# ── 5. Scale, split, train ──────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ── 6. Results ──────────────────────────────────────────────────
print("=" * 50)
print("  BRUTE-FORCE LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"\nTraining set:")
print(f"  R²   = {r2_score(y_train, y_pred_train):.4f}")
print(f"  MAE  = {mean_absolute_error(y_train, y_pred_train):.4f} GHz")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f} GHz")
print(f"\nTest set:")
print(f"  R²   = {r2_score(y_test, y_pred_test):.4f}")
print(f"  MAE  = {mean_absolute_error(y_test, y_pred_test):.4f} GHz")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f} GHz")

# ── 7. Top 10 most influential features ─────────────────────────
print(f"\n--- Top 10 most influential features ---")
importances = pd.DataFrame({
    "Feature": X.columns,
    "Weight": model.coef_
})
importances["Abs Weight"] = importances["Weight"].abs()
importances = importances.sort_values("Abs Weight", ascending=False).head(10)
for _, row in importances.iterrows():
    print(f"  {row['Feature']:35s}  weight = {row['Weight']:+7.4f}")
