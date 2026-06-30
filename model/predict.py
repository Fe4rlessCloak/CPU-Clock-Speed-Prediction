"""
predict.py — one file to rule them all.
Loads consumer Intel data, computes P-Cores inline, engineers features,
trains Ridge, evaluates, then lets you predict future CPUs interactively.

No joblib, no saved weights, no external dependencies beyond pandas/sklearn.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# =========================================================================
# 1.  P-CORES PARSER (ported from add_pcores.py)
# =========================================================================

SKU_OVERRIDES = {
    "12450HX": 4, "1280P": 6, "1370P": 6, "N300": 0, "N305": 0,
    "251HX": 6, "U300": 1, "U300E": 1, "U300L": 1, "U301L": 1,
    "U302L": 1, "U303L": 1, "8500": 1, "8505": 1,
}

E_ONLY_CODE_NAMES = {
    "Alder Lake-N", "Twin Lake", "Jasper Lake", "Elkhart Lake", "Amston Lake",
    "Apollo Lake", "Braswell", "Cherry Trail", "Bay Trail", "Merrifield",
    "Moorefield", "Cloverview", "Cedarview", "Penwell", "SoFIA 3G R",
    "SoFIA LTE", "Gemini Lake", "Gemini Lake Refresh", "Snow Ridge",
    "Parker Ridge", "Hewitt Lake", "Denverton", "Rangeley", "Avoton",
    "Centerton",
}

CODE_NAME_GEN = {
    "Raptor Lake": "13", "Alder Lake": "12", "Meteor Lake": "1",
    "Arrow Lake": "2", "Lunar Lake": "2", "Panther Lake": "3",
}

NEW_TIER_MAP = {"3": "i3", "5": "i5", "7": "i7", "9": "i9"}

PANTHER_LAKE_LOOKUP = {
    "Core Ultra 5 322": 6, "Core Ultra 5 325": 8,
    "Core Ultra 5 332": 6, "Core Ultra 5 335": 8,
    "Core Ultra 5 336H": 4, "Core Ultra 5 338H": 4,
    "Core Ultra 7 355": 8, "Core Ultra 7 356H": 6,
    "Core Ultra 7 365": 8, "Core Ultra 7 366H": 6,
    "Core Ultra 9 386H": 6,
    "Core Ultra X7 358H": 6, "Core Ultra X7 368H": 6,
    "Core Ultra X9 378H": 6, "Core Ultra X9 388H": 6,
}

def _segment_from_suffix(suffix):
    if not suffix: return "S"
    if suffix == "HX": return "HX"
    if suffix[0] == "H": return "H"
    if suffix[0] == "P": return "P"
    if suffix[0] == "U": return "U"
    return "S"

def _core_rule(gen, tier, segment):
    if segment == "S":   return {"i9": 8, "i7": 8, "i5": 6, "i3": 4}.get(tier)
    if segment == "HX":  return {"i9": 8, "i7": 8, "i5": 6}.get(tier)
    if segment == "H":   return {"i9": 6, "i7": 6, "i5": 4, "i3": 2}.get(tier)
    if segment == "P":
        if tier == "i7": return 6 if gen == "14" else 4
        return {"i5": 4, "i3": 2}.get(tier)
    if segment == "U":   return 2
    return None

def _ultra_rule(gen, tier, segment):
    gen = int(gen)
    if gen == 1:
        if segment == "H": return {"Ultra9": 6, "Ultra7": 6, "Ultra5": 4, "Ultra3": 2}.get(tier)
        if segment == "U": return {"Ultra7": 2, "Ultra5": 2, "Ultra3": 2}.get(tier)
    if gen == 2:
        if segment == "V":  return 4
        if segment == "HX": return {"Ultra9": 8, "Ultra7": 8, "Ultra5": 6}.get(tier)
        if segment == "H":  return {"Ultra9": 6, "Ultra7": 6, "Ultra5": 4}.get(tier)
        if segment == "U":  return {"Ultra7": 2, "Ultra5": 2}.get(tier)
        if segment == "S":  return {"Ultra9": 8, "Ultra7": 8, "Ultra5": 6, "Ultra3": 4}.get(tier)
    if gen == 3:
        if segment == "S": return 4
        if segment == "H": return 4
        if segment == "U": return {"Ultra7": 2, "Ultra5": 2}.get(tier)
    return None

def _ultra_segment(suffix):
    if suffix == "V": return "V"
    if suffix in ("HX",): return "HX"
    if suffix[0] == "H": return "H"
    if suffix[0] == "U" or suffix == "UA": return "U"
    return "S"

def get_p_cores(product_name, code_name, total_cores):
    if code_name in E_ONLY_CODE_NAMES:
        return 0

    product_name = re.sub(r"\s*\(.*?\)", "", product_name).strip()

    # Panther Lake explicit lookup (exact match)
    if product_name in PANTHER_LAKE_LOOKUP:
        return PANTHER_LAKE_LOOKUP[product_name]

    # SKU overrides: check longest keys first to avoid substring shadowing
    for sku in sorted(SKU_OVERRIDES, key=len, reverse=True):
        if sku in product_name:
            return SKU_OVERRIDES[sku]

    code_gen = CODE_NAME_GEN.get(code_name, "")

    # Core Ultra
    if "Ultra" in product_name:
        parts = product_name.split()
        if len(parts) < 4:
            return total_cores
        tier_part = parts[2]; num_suffix = parts[3]
        if tier_part.startswith("X"): tier = tier_part[1:]
        else: tier = tier_part
        m = re.match(r"(\d+)([A-Z]*)", num_suffix)
        if not m: return total_cores
        suffix = m.group(2) or "S"
        seg = _ultra_segment(suffix)
        tier_name = f"Ultra{tier}" if tier.isdigit() else tier
        ultra_gen = code_gen if code_gen else m.group(1)[0]
        result = _ultra_rule(ultra_gen, tier_name, seg)
        return result if result is not None else total_cores

    # Core N naming
    m_new = re.match(r"Core\s+(\d+)\s+(\d+)([A-Z]*)", product_name)
    if m_new:
        tier = NEW_TIER_MAP.get(m_new.group(1))
        if not tier: return total_cores
        suffix = m_new.group(3) or "S"
        seg = _segment_from_suffix(suffix)
        gen = code_gen if code_gen else "13"
        result = _core_rule(gen, tier, seg)
        return result if result is not None else total_cores

    # Standard i#-XXXXX
    m = re.search(r"(i\d+)-(\d+)([A-Z]*)", product_name)
    if m:
        tier = m.group(1); suffix = m.group(3) or "S"
        seg = _segment_from_suffix(suffix)
        gen = code_gen if code_gen else m.group(2)[:2]
        if gen in ("12", "13", "14"):
            result = _core_rule(gen, tier, seg)
            if result is not None: return result
        return total_cores

    return total_cores


# =========================================================================
# 2.  LOOKUP TABLES (arch features, node intro, suffix)
# =========================================================================

def classify_suffix(name):
    m = re.search(r'(\d+)([A-Z]+[0-9]?)$', name)
    s = m.group(2) if m else ""
    if s in ("K","KF","KS","X","XE","HK","HX"): return "High Perf"
    elif s in ("H","HQ","HL","HE","QM","MQ"): return "High Perf Mobile"
    elif s in ("U","UM","UL","UE"): return "Ultra-Low Power"
    elif s == "Y": return "Extreme Low Power"
    elif s in ("T","TE"): return "Power Optimized"
    elif s in ("N","NT","NTE"): return "Low Power"
    elif s in ("E","EQ","EC"): return "Embedded"
    elif s in ("F","KF"): return "No Graphics"
    elif s in ("M",): return "Mobile (Legacy)"
    elif s in ("S","P","G","G1","G4","G7"): return "Standard / Graphics"
    elif s in ("B","R","RE","RK"): return "BGA / Soldered"
    else: return "Standard"

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

arch_table = {
    "Core 2": [3072, 0, 0, 4.0], "Nehalem": [256, 0, 0, 4.0],
    "Westmere": [256, 0, 0, 9.0], "Sandy Bridge": [256, 0, 0, 9.0],
    "Ivy Bridge": [256, 0, 0, 17.0], "Haswell": [256, 0, 0, 17.0],
    "Broadwell": [256, 0, 0, 37.5], "Skylake": [256, 0, 0, 37.5],
    "Kaby Lake": [256, 0, 0, 37.5], "Coffee Lake": [256, 0, 0, 37.5],
    "Comet Lake": [256, 0, 0, 37.5], "Whiskey Lake": [256, 0, 0, 37.5],
    "Amber Lake": [256, 0, 0, 37.5], "Ice Lake": [512, 0, 0, 63.0],
    "Tiger Lake": [1280, 0, 0, 63.0], "Rocket Lake": [512, 0, 0, 37.5],
    "Alder Lake": [1280, 0, 0, 100.0], "Raptor Lake": [2048, 0, 0, 100.0],
    "Meteor Lake": [2048, 1, 0, 100.0], "Lunar Lake": [2560, 1, 0, 170.0],
    "Arrow Lake": [3072, 1, 0, 170.0], "Panther Lake": [3072, 1, 0, 150.0],
    "Xeon Scalable": [2048, 1, 1, 100.0], "Atom": [512, 0, 0, 100.0],
    "Hybrid": [512, 1, 0, 63.0], "Xeon Legacy": [256, 0, 0, 17.0],
}

node_intro = {
    "Core 2": 2007, "Nehalem": 2007, "Westmere": 2010, "Sandy Bridge": 2010,
    "Ivy Bridge": 2012, "Haswell": 2012, "Broadwell": 2014, "Skylake": 2014,
    "Kaby Lake": 2014, "Coffee Lake": 2014, "Comet Lake": 2014,
    "Whiskey Lake": 2014, "Amber Lake": 2014, "Rocket Lake": 2014,
    "Ice Lake": 2019, "Tiger Lake": 2019, "Alder Lake": 2019,
    "Raptor Lake": 2019, "Atom": 2019, "Hybrid": 2019, "Meteor Lake": 2023,
    "Lunar Lake": 2024, "Arrow Lake": 2024, "Panther Lake": 2025,
    "Xeon Scalable": 2017, "Xeon Legacy": 2012,
}

family_map = {
    "Atom": "Atom", "Celeron": "Celeron", "Core": "Core",
    "CoreUltra": "Core Ultra", "Intel": "Intel", "Pentium": "Pentium",
}


# =========================================================================
# 3.  DATA LOADING + FEATURE ENGINEERING
# =========================================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
    d = pd.read_csv(f)
    d["Vertical Segment"] = f.stem.split("-")[0]
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df["Vertical Segment"] = df["Vertical Segment"].replace(family_map)

# ---- Compute P-Cores inline ----
df["P-Cores"] = df.apply(
    lambda r: get_p_cores(str(r["Product"]), str(r["Code Name"]),
                          int(float(r["Cores"])) if pd.notna(r["Cores"]) else 0),
    axis=1
)

# ---- Base features ----
df["Release Year"] = df["Release Date"].str[-2:].astype(int) + 2000
df["Power Tier"] = df["Product"].apply(classify_suffix)
df["Arch Group"] = df["Code Name"].map(gen_map)

for grp, vals in arch_table.items():
    mask = df["Arch Group"] == grp
    df.loc[mask, "L2_per_Core_KB"] = vals[0]
    df.loc[mask, "Is_Tiled"] = vals[1]
    df.loc[mask, "Is_Mesh"] = vals[2]
    df.loc[mask, "Node_Density"] = vals[3]

for c in ["Lithography(nm)", "Cores", "TDP(W)", "Threads",
          "L2_per_Core_KB", "Is_Tiled", "Is_Mesh", "Node_Density"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---- Engineered features ----
df["Log_Node_Density"] = np.log1p(df["Node_Density"])
df["Cores_x_Is_Mesh"] = df["Cores"] * df["Is_Mesh"]
df["TDP_x_Is_Tiled"] = df["TDP(W)"] * df["Is_Tiled"]
df["Node_Intro_Year"] = df["Arch Group"].map(node_intro)
df["Node_Maturity_Years"] = df["Release Year"] - df["Node_Intro_Year"]
df["TDP_per_Core"] = df["TDP(W)"] / df["Cores"]
df["Threads_per_Core"] = df["Threads"] / df["Cores"]
df["Power_Starvation_Index"] = (df["Cores"] ** 2) / df["TDP(W)"]

# P-Cores features with fallback: when P-Cores = 0, use total Cores
p_cores_fallback = df["P-Cores"].copy()
zero_mask = p_cores_fallback == 0
p_cores_fallback[zero_mask] = df["Cores"][zero_mask]
df["P_Core_Ratio"] = p_cores_fallback / df["Cores"]   # 1.0 for E-only chips
df["TDP_per_PCore"] = df["TDP(W)"] / p_cores_fallback

df.drop(columns=["Node_Density", "Node_Intro_Year"], inplace=True)

y = pd.to_numeric(df["Max. Turbo Freq.(GHz)"], errors="coerce")

# ---- One-hot encode ----
family_dummies = pd.get_dummies(df["Vertical Segment"], prefix="fam", drop_first=True)
tier_dummies = pd.get_dummies(df["Power Tier"], prefix="tier", drop_first=True)

base_features = [
    "Lithography(nm)", "Cores", "TDP(W)", "Release Year",
    "L2_per_Core_KB", "Is_Tiled", "Is_Mesh", "Log_Node_Density",
    "Cores_x_Is_Mesh", "TDP_x_Is_Tiled",
    "Node_Maturity_Years", "TDP_per_Core", "Threads_per_Core",
    "Power_Starvation_Index", "P_Core_Ratio", "TDP_per_PCore",
]

X = pd.concat([df[base_features], family_dummies, tier_dummies], axis=1)

mask = X.isna().any(axis=1) | y.isna()
X = X[~mask]
y = y[~mask]

print(f"Training samples: {len(X)}")

# ---- Scale, split, train ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=628.03)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
print(f"Training samples: {len(X)}")
print(f"Poly features   : {X_poly.shape[1]}")
print(f"Test R-squared  = {r2_score(y_test, y_pred_test):.4f}")
print(f"Test MAE        = {mean_absolute_error(y_test, y_pred_test):.4f} GHz")
print()

all_columns = X.columns.tolist()
family_dummy_cols = family_dummies.columns.tolist()
tier_dummy_cols = tier_dummies.columns.tolist()


# =========================================================================
# 4.  INTERACTIVE PREDICTION LOOP
# =========================================================================

while True:
    print("=" * 60)
    print("  FUTURE CPU PREDICTOR  (enter 0 to quit)")
    print("=" * 60)

    lithography = float(input("  Lithography(nm)       [e.g. 3]: ") or 0)
    if lithography == 0: break
    cores = float(input("  Cores                 [e.g. 8]: "))
    threads = float(input("  Threads               [e.g. 16]: "))
    tdp = float(input("  TDP(W)                [e.g. 125]: "))
    release_year = float(input("  Release Year          [e.g. 2026]: "))
    l2_per_core = float(input("  L2 per P-core (KB)    [e.g. 3072]: "))
    is_tiled = int(input("  Is Tiled? (0/1)       [e.g. 1]: "))
    is_mesh = int(input("  Is Mesh?  (0/1)       [e.g. 0]: "))
    node_density = float(input("  Node Density (MTr/mm2) [e.g. 200]: "))
    node_maturity = float(input("  Node Maturity (years)  [e.g. 2]: "))
    p_cores = float(input("  P-Cores               [e.g. 6]: "))

    print("  Family options: Atom, Celeron, Core, Core Ultra, Intel, Pentium")
    family = input("  Family: ").strip()
    print("  Tier options: Standard, High Perf, High Perf Mobile, Ultra-Low Power,...")
    tier = input("  Power Tier: ").strip()

    log_nd = np.log1p(node_density)
    p_safe = p_cores if p_cores > 0 else cores  # fallback to Cores when P=0
    fd = {
        "Lithography(nm)": lithography, "Cores": cores, "TDP(W)": tdp,
        "Release Year": release_year, "L2_per_Core_KB": l2_per_core,
        "Is_Tiled": is_tiled, "Is_Mesh": is_mesh,
        "Log_Node_Density": log_nd,
        "Cores_x_Is_Mesh": cores * is_mesh,
        "TDP_x_Is_Tiled": tdp * is_tiled,
        "Node_Maturity_Years": node_maturity,
        "TDP_per_Core": tdp / cores,
        "Threads_per_Core": threads / cores,
        "Power_Starvation_Index": (cores ** 2) / tdp,
        "P_Core_Ratio": p_safe / cores,
        "TDP_per_PCore": tdp / p_safe,
    }
    for col in family_dummy_cols:
        d = col.replace("fam_", "").replace("_", " ").strip()
        fd[col] = 1 if family.lower() == d.lower() else 0
    for col in tier_dummy_cols:
        d = col.replace("tier_", "").replace("_", " ").strip()
        fd[col] = 1 if tier.lower() == d.lower() else 0

    inp = pd.DataFrame([fd])[all_columns]
    if inp.isna().any().any():
        print("\n  WARNING: NaN detected in input — check your values.")
        print(inp.isna().sum())
    else:
        pred = model.predict(poly.transform(scaler.transform(inp)))[0]
        print(f"\n  >>> Predicted Max. Turbo Freq.:  {pred:.3f} GHz  (~{pred*1000:.0f} MHz)")
    print()
