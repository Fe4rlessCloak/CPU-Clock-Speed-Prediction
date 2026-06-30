"""
predict_bench.py — CPU Turbo Frequency Predictor + Benchmark Suite
===================================================================
Loads & trains the Ridge model (same logic as predict.py), then
evaluates it against a hand-curated list of 10 real CPUs.

EDITING THE BENCHMARK LIST
  → Find the BENCHMARK_CPUS list below (section 5).
    Each entry is a plain dict — add/remove/tweak freely.

RUNNING
  python predict_bench.py  [--data /path/to/data/dir]

If --data is omitted the script tries ../data/ relative to this file.
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

# =========================================================================
# 1.  P-CORES PARSER
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
    if not suffix:        return "S"
    if suffix == "HX":   return "HX"
    if suffix[0] == "H": return "H"
    if suffix[0] == "P": return "P"
    if suffix[0] == "U": return "U"
    return "S"


def _core_rule(gen, tier, segment):
    if segment == "S":  return {"i9": 8, "i7": 8, "i5": 6, "i3": 4}.get(tier)
    if segment == "HX": return {"i9": 8, "i7": 8, "i5": 6}.get(tier)
    if segment == "H":  return {"i9": 6, "i7": 6, "i5": 4, "i3": 2}.get(tier)
    if segment == "P":
        if tier == "i7": return 6 if gen == "14" else 4
        return {"i5": 4, "i3": 2}.get(tier)
    if segment == "U":  return 2
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
        if segment == "S":  return 4
        if segment == "H":  return 4
        if segment == "U":  return {"Ultra7": 2, "Ultra5": 2}.get(tier)
    return None


def _ultra_segment(suffix):
    if suffix == "V":       return "V"
    if suffix in ("HX",):  return "HX"
    if suffix[0] == "H":   return "H"
    if suffix[0] == "U" or suffix == "UA": return "U"
    return "S"


def get_p_cores(product_name, code_name, total_cores):
    if code_name in E_ONLY_CODE_NAMES:
        return 0

    product_name = re.sub(r"\s*\(.*?\)", "", product_name).strip()

    if product_name in PANTHER_LAKE_LOOKUP:
        return PANTHER_LAKE_LOOKUP[product_name]

    for sku in sorted(SKU_OVERRIDES, key=len, reverse=True):
        if sku in product_name:
            return SKU_OVERRIDES[sku]

    code_gen = CODE_NAME_GEN.get(code_name, "")

    if "Ultra" in product_name:
        parts = product_name.split()
        if len(parts) < 4:
            return total_cores
        tier_part = parts[2]
        num_suffix = parts[3]
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

    m_new = re.match(r"Core\s+(\d+)\s+(\d+)([A-Z]*)", product_name)
    if m_new:
        tier = NEW_TIER_MAP.get(m_new.group(1))
        if not tier: return total_cores
        suffix = m_new.group(3) or "S"
        seg = _segment_from_suffix(suffix)
        gen = code_gen if code_gen else "13"
        result = _core_rule(gen, tier, seg)
        return result if result is not None else total_cores

    m = re.search(r"(i\d+)-(\d+)([A-Z]*)", product_name)
    if m:
        tier = m.group(1)
        suffix = m.group(3) or "S"
        seg = _segment_from_suffix(suffix)
        gen = code_gen if code_gen else m.group(2)[:2]
        if gen in ("12", "13", "14"):
            result = _core_rule(gen, tier, seg)
            if result is not None: return result
        return total_cores

    return total_cores


# =========================================================================
# 2.  LOOKUP TABLES
# =========================================================================

def classify_suffix(name):
    m = re.search(r"(\d+)([A-Z]+[0-9]?)$", name)
    s = m.group(2) if m else ""
    if s in ("K", "KF", "KS", "X", "XE", "HK", "HX"):   return "High Perf"
    elif s in ("H", "HQ", "HL", "HE", "QM", "MQ"):       return "High Perf Mobile"
    elif s in ("U", "UM", "UL", "UE"):                    return "Ultra-Low Power"
    elif s == "Y":                                         return "Extreme Low Power"
    elif s in ("T", "TE"):                                 return "Power Optimized"
    elif s in ("N", "NT", "NTE"):                         return "Low Power"
    elif s in ("E", "EQ", "EC"):                          return "Embedded"
    elif s in ("F", "KF"):                                 return "No Graphics"
    elif s in ("M",):                                     return "Mobile (Legacy)"
    elif s in ("S", "P", "G", "G1", "G4", "G7"):         return "Standard / Graphics"
    elif s in ("B", "R", "RE", "RK"):                     return "BGA / Soldered"
    else:                                                  return "Standard"


gen_map = {
    "Merom": "Core 2", "Conroe": "Core 2", "Kentsfield": "Core 2",
    "Wolfdale": "Core 2", "Yorkfield": "Core 2", "Penryn": "Core 2",
    "Bloomfield": "Nehalem", "Lynnfield": "Nehalem",
    "Nehalem EP": "Nehalem", "Nehalem EX": "Nehalem",
    "Gulftown": "Westmere", "Clarkdale": "Westmere",
    "Arrandale": "Westmere", "Westmere EP": "Westmere",
    "Westmere EX": "Westmere",
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
    "Sapphire Rapids": "Xeon Scalable", "Emerald Rapids": "Xeon Scalable",
    "Cascade Lake": "Xeon Scalable", "Cooper Lake": "Xeon Scalable",
    "Denverton": "Atom", "Rangeley": "Atom", "Avoton": "Atom",
    "Centerton": "Atom", "Braswell": "Atom", "Cherry Trail": "Atom",
    "Bay Trail": "Atom", "Merrifield": "Atom", "Moorefield": "Atom",
    "Jasper Lake": "Atom", "Elkhart Lake": "Atom", "Amston Lake": "Atom",
    "Cloverview": "Atom", "Cedarview": "Atom", "Penwell": "Atom",
    "SoFIA 3G R": "Atom", "SoFIA LTE": "Atom", "Twin Lake": "Atom",
    "Gemini Lake": "Atom", "Gemini Lake Refresh": "Atom",
    "Snow Ridge": "Atom", "Parker Ridge": "Atom", "Hewitt Lake": "Atom",
    "Lakefield": "Hybrid",
    "Jasper Forest": "Xeon Legacy", "Nocona": "Xeon Legacy",
    "Crystal Well": "Haswell", "Devil's Canyon": "Haswell",
}

arch_table = {
    "Core 2":       [3072, 0, 0,  4.0],
    "Nehalem":      [ 256, 0, 0,  4.0],
    "Westmere":     [ 256, 0, 0,  9.0],
    "Sandy Bridge": [ 256, 0, 0,  9.0],
    "Ivy Bridge":   [ 256, 0, 0, 17.0],
    "Haswell":      [ 256, 0, 0, 17.0],
    "Broadwell":    [ 256, 0, 0, 37.5],
    "Skylake":      [ 256, 0, 0, 37.5],
    "Kaby Lake":    [ 256, 0, 0, 37.5],
    "Coffee Lake":  [ 256, 0, 0, 37.5],
    "Comet Lake":   [ 256, 0, 0, 37.5],
    "Whiskey Lake": [ 256, 0, 0, 37.5],
    "Amber Lake":   [ 256, 0, 0, 37.5],
    "Ice Lake":     [ 512, 0, 0, 63.0],
    "Tiger Lake":   [1280, 0, 0, 63.0],
    "Rocket Lake":  [ 512, 0, 0, 37.5],
    "Alder Lake":   [1280, 0, 0,100.0],
    "Raptor Lake":  [2048, 0, 0,100.0],
    "Meteor Lake":  [2048, 1, 0,100.0],
    "Lunar Lake":   [2560, 1, 0,170.0],
    "Arrow Lake":   [3072, 1, 0,170.0],
    "Panther Lake": [3072, 1, 0,150.0],
    "Xeon Scalable":[2048, 1, 1,100.0],
    "Atom":         [ 512, 0, 0,100.0],
    "Hybrid":       [ 512, 1, 0, 63.0],
    "Xeon Legacy":  [ 256, 0, 0, 17.0],
}

node_intro = {
    "Core 2": 2007, "Nehalem": 2007, "Westmere": 2010,
    "Sandy Bridge": 2010, "Ivy Bridge": 2012, "Haswell": 2012,
    "Broadwell": 2014, "Skylake": 2014, "Kaby Lake": 2014,
    "Coffee Lake": 2014, "Comet Lake": 2014, "Whiskey Lake": 2014,
    "Amber Lake": 2014, "Rocket Lake": 2014, "Ice Lake": 2019,
    "Tiger Lake": 2019, "Alder Lake": 2019, "Raptor Lake": 2019,
    "Atom": 2019, "Hybrid": 2019, "Meteor Lake": 2023,
    "Lunar Lake": 2024, "Arrow Lake": 2024, "Panther Lake": 2025,
    "Xeon Scalable": 2017, "Xeon Legacy": 2012,
}

family_map = {
    "Atom": "Atom", "Celeron": "Celeron", "Core": "Core",
    "CoreUltra": "Core Ultra", "Intel": "Intel", "Pentium": "Pentium",
}

BASE_FEATURES = [
    "Lithography(nm)", "Cores", "TDP(W)", "Release Year",
    "L2_per_Core_KB", "Is_Tiled", "Is_Mesh", "Log_Node_Density",
    "Cores_x_Is_Mesh", "TDP_x_Is_Tiled",
    "Node_Maturity_Years", "TDP_per_Core", "Threads_per_Core",
    "Power_Starvation_Index", "P_Core_Ratio", "TDP_per_PCore",
]


# =========================================================================
# 3.  DATA LOADING + FEATURE ENGINEERING
# =========================================================================

def load_and_engineer(data_dir: Path):
    csv_files = [
        data_dir / "Core-Processors-1.8.csv",
        data_dir / "Atom-Processors-1.9-16col.csv",
        data_dir / "Celeron-Processors-1.9-16col.csv",
        data_dir / "Core_Ultra-Processors-1.10-16col.csv",
        data_dir / "Intel-Processors-1.9-16col.csv",
        data_dir / "Pentium-Processors-1.9-16col.csv",
    ]

    frames = []
    for f in csv_files:
        if not f.exists():
            print(f"  [WARN] Missing: {f.name}")
            continue
        d = pd.read_csv(f)
        d["Vertical Segment"] = f.stem.split("-")[0]
        frames.append(d)

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Pass --data /path/to/data or place CSVs in ../data/"
        )

    df = pd.concat(frames, ignore_index=True)
    df["Vertical Segment"] = df["Vertical Segment"].replace(family_map)

    df["P-Cores"] = df.apply(
        lambda r: get_p_cores(
            str(r["Product"]), str(r["Code Name"]),
            int(float(r["Cores"])) if pd.notna(r["Cores"]) else 0,
        ),
        axis=1,
    )

    df["Release Year"] = df["Release Date"].str[-2:].astype(int) + 2000
    df["Power Tier"] = df["Product"].apply(classify_suffix)
    df["Arch Group"] = df["Code Name"].map(gen_map)

    for grp, vals in arch_table.items():
        mask = df["Arch Group"] == grp
        df.loc[mask, "L2_per_Core_KB"] = vals[0]
        df.loc[mask, "Is_Tiled"]       = vals[1]
        df.loc[mask, "Is_Mesh"]        = vals[2]
        df.loc[mask, "Node_Density"]   = vals[3]

    for c in ["Lithography(nm)", "Cores", "TDP(W)", "Threads",
              "L2_per_Core_KB", "Is_Tiled", "Is_Mesh", "Node_Density"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Log_Node_Density"]       = np.log1p(df["Node_Density"])
    df["Cores_x_Is_Mesh"]        = df["Cores"] * df["Is_Mesh"]
    df["TDP_x_Is_Tiled"]         = df["TDP(W)"] * df["Is_Tiled"]
    df["Node_Intro_Year"]        = df["Arch Group"].map(node_intro)
    df["Node_Maturity_Years"]    = df["Release Year"] - df["Node_Intro_Year"]
    df["TDP_per_Core"]           = df["TDP(W)"] / df["Cores"]
    df["Threads_per_Core"]       = df["Threads"] / df["Cores"]
    df["Power_Starvation_Index"] = (df["Cores"] ** 2) / df["TDP(W)"]

    # P-Cores features: when P=0 fall back to total Cores
    df["P_Core_Ratio"] = np.where(
        df["P-Cores"] == 0,
        1.0,
        df["P-Cores"] / df["Cores"]
    )
    df["TDP_per_PCore"] = np.where(
        df["P-Cores"] == 0,
        df["TDP(W)"] / df["Cores"],
        df["TDP(W)"] / df["P-Cores"]
    )

    df.drop(columns=["Node_Density", "Node_Intro_Year"], inplace=True)
    return df


def build_model(df):
    y = pd.to_numeric(df["Max. Turbo Freq.(GHz)"], errors="coerce")

    family_dummies = pd.get_dummies(df["Vertical Segment"], prefix="fam", drop_first=True)
    tier_dummies   = pd.get_dummies(df["Power Tier"],       prefix="tier", drop_first=True)

    X = pd.concat([df[BASE_FEATURES], family_dummies, tier_dummies], axis=1)

    mask = X.isna().any(axis=1) | y.isna()
    X, y = X[~mask], y[~mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)

    X_tr, X_te, y_tr, y_te = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=628.03)
    model.fit(X_tr, y_tr)

    y_pred_te = model.predict(X_te)
    print(f"  Training samples : {len(X)}")
    print(f"  Poly features    : {X_poly.shape[1]}")
    print(f"  Test R²          : {r2_score(y_te, y_pred_te):.4f}")
    print(f"  Test MAE         : {mean_absolute_error(y_te, y_pred_te):.4f} GHz")

    return model, poly, scaler, X.columns.tolist(), family_dummies.columns.tolist(), tier_dummies.columns.tolist()


# =========================================================================
# 4.  FEATURE BUILDER (dict → DataFrame row)
# =========================================================================

def build_row(cpu, all_cols, family_cols, tier_cols):
    """Convert a CPU benchmark dict into a scaled-ready feature row."""
    nd   = cpu["node_density"]
    p    = cpu["p_cores"]
    c    = cpu["cores"]
    tdp  = cpu["tdp"]
    thr  = cpu["threads"]

    p_ratio = 1.0 if p == 0 else (p / c)       # fall back to Cores when P=0
    tdp_p   = (tdp / c) if p == 0 else (tdp / p)  # fall back to TDP_per_Core
    fd = {
        "Lithography(nm)":        cpu["lithography"],
        "Cores":                  c,
        "TDP(W)":                 tdp,
        "Release Year":           cpu["release_year"],
        "L2_per_Core_KB":         cpu["l2_per_core_kb"],
        "Is_Tiled":               cpu["is_tiled"],
        "Is_Mesh":                cpu["is_mesh"],
        "Log_Node_Density":       np.log1p(nd),
        "Cores_x_Is_Mesh":        c * cpu["is_mesh"],
        "TDP_x_Is_Tiled":         tdp * cpu["is_tiled"],
        "Node_Maturity_Years":    cpu["node_maturity_years"],
        "TDP_per_Core":           tdp / c,
        "Threads_per_Core":       thr / c,
        "Power_Starvation_Index": (c ** 2) / tdp,
        "P_Core_Ratio":           p_ratio, 
        "TDP_per_PCore":          tdp_p,    
    }

    for col in family_cols:
        fam_label = col.replace("fam_", "").replace("_", " ").strip()
        fd[col] = 1 if cpu["family"].lower() == fam_label.lower() else 0

    for col in tier_cols:
        tier_label = col.replace("tier_", "").replace("_", " ").strip()
        fd[col] = 1 if cpu["power_tier"].lower() == tier_label.lower() else 0

    return pd.DataFrame([fd])[all_cols]


# =========================================================================
# 5.  BENCHMARK SUITE  ← EDIT THIS LIST TO ADD/REMOVE/CHANGE CPUS
# =========================================================================
# Keys explained:
#   name              — human label (not used in prediction, just display)
#   actual_ghz        — real Intel spec (Max Turbo); set to None to skip error calc
#   lithography       — process node in nm
#   cores             — total core count (P + E)
#   p_cores           — Performance cores only  (0 for E-only Atoms)
#   threads           — logical threads
#   tdp               — PL1 / TDP in watts
#   release_year      — year the CPU launched
#   l2_per_core_kb    — L2 cache per P-core in KB  (use arch_table values above)
#   is_tiled          — 1 if chiplet/tile design, else 0
#   is_mesh           — 1 if Mesh interconnect (Xeon-class), else 0
#   node_density      — transistor density in MTr/mm² (from arch_table)
#   node_maturity_years — release_year minus node_intro year
#   family            — "Core", "Core Ultra", "Atom", "Celeron", "Pentium", "Intel"
#   power_tier        — from classify_suffix():
#                       "High Perf" | "High Perf Mobile" | "Ultra-Low Power" |
#                       "Extreme Low Power" | "Power Optimized" | "Low Power" |
#                       "Embedded" | "No Graphics" | "Standard"
#
# WHY THESE 10?  Each one stresses a different corner of the feature space:
#   i9-13900K   → many-core, overclockable, mature node
#   i9-12900HX  → mobile HX, hybrid design debut
#   i3-1215U    → entry Alder, low TDP, mostly E-cores
#   Core Ultra 9 185H → Meteor Lake tiled debut, hard to generalise
#   i5-1335U    → Raptor refresh U-class, narrow TDP window
#   i9-14900KS  → highest SKU binned part, extreme turbo
#   Celeron N4500 → pure E-core Jasper Lake, zero P-cores
#   Core Ultra 7 165H → Arrow Lake predecessor, reference mobile
#   i5-10600K   → Comet Lake, pre-hybrid, single-tile
#   i7-1260P    → 12th-gen P-segment (mixed core, moderate TDP)

BENCHMARK_CPUS = [
    # =====================================================================
    # VERY OLD (Nehalem, Sandy Bridge, Ivy Bridge)
    # =====================================================================
    {
        "name": "Intel Core i7-920",
        "actual_ghz": 2.93,
        "lithography": 45,
        "cores": 4,
        "p_cores": 4,                         # All P-cores
        "threads": 8,
        "tdp": 130,
        "release_year": 2008,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 10.0,                 # 45nm
        "node_maturity_years": 1,             # 45nm introduced 2007
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i7-2700K",
        "actual_ghz": 3.9,
        "lithography": 32,
        "cores": 4,
        "p_cores": 4,
        "threads": 8,
        "tdp": 95,
        "release_year": 2011,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 15.0,                 # 32nm
        "node_maturity_years": 1,             # 32nm introduced 2010
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i7-3770K",
        "actual_ghz": 3.9,
        "lithography": 22,
        "cores": 4,
        "p_cores": 4,
        "threads": 8,
        "tdp": 77,
        "release_year": 2012,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 25.0,                 # 22nm
        "node_maturity_years": 0,             # 22nm introduced 2012
        "family": "Core",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # HASWELL & BROADWELL (with eDRAM!)
    # =====================================================================
    {
        "name": "Intel Core i7-4790K",
        "actual_ghz": 4.4,
        "lithography": 22,
        "cores": 4,
        "p_cores": 4,
        "threads": 8,
        "tdp": 88,
        "release_year": 2014,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 25.0,
        "node_maturity_years": 2,             # 22nm introduced 2012
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i7-5775C",
        "actual_ghz": 3.7,
        "lithography": 14,
        "cores": 4,
        "p_cores": 4,
        "threads": 8,
        "tdp": 65,
        "release_year": 2015,
        "l2_per_core_kb": 256,                # 1MB L2 total, plus 128MB L4 eDRAM
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,                 # 14nm
        "node_maturity_years": 0,             # 14nm introduced 2015
        "family": "Core",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # SKYLAKE, KABY LAKE, COFFEE LAKE (Common & Extreme)
    # =====================================================================
    {
        "name": "Intel Core i7-6700K",
        "actual_ghz": 4.2,
        "lithography": 14,
        "cores": 4,
        "p_cores": 4,
        "threads": 8,
        "tdp": 91,
        "release_year": 2015,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 0,
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i7-7700K",
        "actual_ghz": 4.5,
        "lithography": 14,
        "cores": 4,
        "p_cores": 4,
        "threads": 8,
        "tdp": 91,
        "release_year": 2017,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 2,             # 14nm introduced 2015
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i5-8400",
        "actual_ghz": 4.0,
        "lithography": 14,
        "cores": 6,
        "p_cores": 6,
        "threads": 6,
        "tdp": 65,
        "release_year": 2017,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 2,
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i9-9900K",
        "actual_ghz": 5.0,
        "lithography": 14,
        "cores": 8,
        "p_cores": 8,
        "threads": 16,
        "tdp": 95,
        "release_year": 2018,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 3,
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i9-10900K",
        "actual_ghz": 5.3,
        "lithography": 14,
        "cores": 10,
        "p_cores": 10,
        "threads": 20,
        "tdp": 125,
        "release_year": 2020,
        "l2_per_core_kb": 256,                # 2.5MB L2 total
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 5,             # 14nm introduced 2015
        "family": "Core",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # ROCKET LAKE (14nm backport, larger L2)
    # =====================================================================
    {
        "name": "Intel Core i7-11700K",
        "actual_ghz": 5.0,
        "lithography": 14,
        "cores": 8,
        "p_cores": 8,
        "threads": 16,
        "tdp": 125,
        "release_year": 2021,
        "l2_per_core_kb": 512,                # 4MB L2 total
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 6,
        "family": "Core",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # ALDER LAKE (12th Gen) – Budget, Common, High-End
    # =====================================================================
    {
        "name": "Intel Core i3-12100F",
        "actual_ghz": 4.3,
        "lithography": 10,                    # Intel 7
        "cores": 4,
        "p_cores": 4,                         # All P-cores, no E-cores
        "threads": 8,
        "tdp": 58,
        "release_year": 2022,
        "l2_per_core_kb": 1280,               # 5MB L2 total
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 1,             # Intel 7 introduced 2021
        "family": "Core",
        "power_tier": "Standard / Graphics",
    },
    {
        "name": "Intel Core i5-12400",
        "actual_ghz": 4.4,
        "lithography": 10,
        "cores": 6,
        "p_cores": 6,                         # All P-cores, no E-cores
        "threads": 12,
        "tdp": 65,
        "release_year": 2022,
        "l2_per_core_kb": 1280,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 1,
        "family": "Core",
        "power_tier": "Standard / Graphics",
    },
    {
        "name": "Intel Core i7-12700K",
        "actual_ghz": 5.0,
        "lithography": 10,
        "cores": 12,                          # 8P + 4E
        "p_cores": 8,
        "threads": 20,
        "tdp": 125,
        "release_year": 2021,
        "l2_per_core_kb": 1194,               # 14MB L2 total / 12
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 0,
        "family": "Core",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # RAPTOR LAKE (13th/14th Gen) – Common Gaming & Extreme
    # =====================================================================
    {
        "name": "Intel Core i5-13600K",
        "actual_ghz": 5.1,
        "lithography": 10,
        "cores": 14,                          # 6P + 8E
        "p_cores": 6,
        "threads": 20,
        "tdp": 125,
        "release_year": 2022,
        "l2_per_core_kb": 1463,               # 20.5MB L2 total / 14 (rounded)
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 1,
        "family": "Core",
        "power_tier": "High Perf",
    },
    {
        "name": "Intel Core i9-14900KS",
        "actual_ghz": 6.2,
        "lithography": 10,
        "cores": 24,                          # 8P + 16E
        "p_cores": 8,
        "threads": 32,
        "tdp": 150,
        "release_year": 2024,
        "l2_per_core_kb": 1365,               # 32.75MB L2 total / 24
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 3,
        "family": "Core",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # CORE ULTRA – METEOR LAKE (Series 1) – Mobile & Ultra-Low
    # =====================================================================
    {
        "name": "Intel Core Ultra 7 155U",
        "actual_ghz": 4.8,
        "lithography": 7,                     # Intel 4
        "cores": 12,                          # 2P + 8E + 2LPE
        "p_cores": 2,
        "threads": 14,                        # Only P-cores have HT
        "tdp": 15,
        "release_year": 2023,
        "l2_per_core_kb": 1024,               # ~12.3MB L2 total / 12
        "is_tiled": 1,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 0,
        "family": "Core Ultra",
        "power_tier": "Ultra-Low Power",
    },
    {
        "name": "Intel Core Ultra 5 125H",
        "actual_ghz": 4.5,
        "lithography": 7,
        "cores": 14,                          # 4P + 8E + 2LPE
        "p_cores": 4,
        "threads": 18,
        "tdp": 28,
        "release_year": 2023,
        "l2_per_core_kb": 1286,               # 18MB L2 total / 14
        "is_tiled": 1,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 0,
        "family": "Core Ultra",
        "power_tier": "High Perf Mobile",
    },

    # =====================================================================
    # CORE ULTRA – ARROW LAKE (Series 2) – Desktop Extreme
    # =====================================================================
    {
        "name": "Intel Core Ultra 9 285K",
        "actual_ghz": 5.7,
        "lithography": 3,                     # TSMC N3B
        "cores": 24,                          # 8P + 16E
        "p_cores": 8,
        "threads": 24,                        # No hyperthreading
        "tdp": 125,
        "release_year": 2024,
        "l2_per_core_kb": 1707,               # 41MB L2 total / 24
        "is_tiled": 1,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 0,
        "family": "Core Ultra",
        "power_tier": "High Perf",
    },

    # =====================================================================
    # CORE ULTRA – LUNAR LAKE (Series 2) – Niche Ultra-Low
    # =====================================================================
    {
        "name": "Intel Core Ultra 7 258V",
        "actual_ghz": 4.8,
        "lithography": 3,
        "cores": 8,                           # 4P + 4E
        "p_cores": 4,
        "threads": 8,                         # No hyperthreading
        "tdp": 17,
        "release_year": 2024,
        "l2_per_core_kb": 1792,               # 14MB L2 total / 8
        "is_tiled": 1,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 0,
        "family": "Core Ultra",
        "power_tier": "Ultra-Low Power",
    },

    # =====================================================================
    # NICHE – 1 P-core Hybrid (U300 / Pentium Gold)
    # =====================================================================
    {
        "name": "Intel Pentium Gold 8505",
        "actual_ghz": 4.4,
        "lithography": 10,
        "cores": 5,                           # 1P + 4E
        "p_cores": 1,
        "threads": 6,
        "tdp": 15,
        "release_year": 2022,
        "l2_per_core_kb": 1024,               # ~5MB L2 total / 5
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 1,
        "family": "Pentium",
        "power_tier": "Ultra-Low Power",
    },

    # =====================================================================
    # NICHE – PURE E-core (Alder Lake-N / Jasper Lake)
    # =====================================================================
    {
        "name": "Intel Processor N95",
        "actual_ghz": 3.4,
        "lithography": 10,
        "cores": 4,
        "p_cores": 0,                         # 100% E-cores
        "threads": 4,
        "tdp": 15,
        "release_year": 2023,
        "l2_per_core_kb": 512,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 2,
        "family": "Processor",
        "power_tier": "Low Power",
    },
    {
        "name": "Intel Core i3-N300",
        "actual_ghz": 3.8,
        "lithography": 10,
        "cores": 8,                           # 8 E-cores
        "p_cores": 0,
        "threads": 8,
        "tdp": 7,
        "release_year": 2023,
        "l2_per_core_kb": 512,                # 4MB L2 total / 8
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 2,
        "family": "Core",
        "power_tier": "Low Power",
    },
    {
        "name": "Intel Pentium Silver N6005",
        "actual_ghz": 3.3,
        "lithography": 10,
        "cores": 4,
        "p_cores": 0,
        "threads": 4,
        "tdp": 10,
        "release_year": 2021,
        "l2_per_core_kb": 384,                # 1.5MB L2 total / 4
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 100.0,
        "node_maturity_years": 0,
        "family": "Pentium",
        "power_tier": "Low Power",
    },

    # =====================================================================
    # NICHE – FANLESS / Y-SERIES (Core m3)
    # =====================================================================
    {
        "name": "Intel Core m3-7Y30",
        "actual_ghz": 2.6,
        "lithography": 14,
        "cores": 2,
        "p_cores": 2,
        "threads": 4,
        "tdp": 4.5,
        "release_year": 2016,
        "l2_per_core_kb": 256,
        "is_tiled": 0,
        "is_mesh": 0,
        "node_density": 37.5,
        "node_maturity_years": 1,
        "family": "Core",
        "power_tier": "Ultra-Low Power",
    },

    # =====================================================================
    # EXTREME – OVERCLOCKING LEGENDS (additional)
    # =====================================================================
    {
        "name": "Intel Core i7-6950X",
        "actual_ghz": 4.0,
        "lithography": 14,
        "cores": 10,                          # All P-cores (HEDT)
        "p_cores": 10,
        "threads": 20,
        "tdp": 140,
        "release_year": 2016,
        "l2_per_core_kb": 256,                # 2.5MB L2 total
        "is_tiled": 0,
        "is_mesh": 1,                         # Broadwell-E uses mesh
        "node_density": 37.5,
        "node_maturity_years": 1,
        "family": "Core",
        "power_tier": "High Perf",
    },
]



# =========================================================================
# 6.  BENCHMARK RUNNER
# =========================================================================

DIVIDER = "─" * 72

def run_benchmarks(model, poly, scaler, all_cols, fam_cols, tier_cols):
    print(f"\n{'═' * 72}")
    print("  BENCHMARK RESULTS")
    print(f"{'═' * 72}")
    print(
        f"  {'CPU':<45} {'Actual':>7} {'Pred':>7} {'Err':>7} {'%Err':>6}"
    )
    print(DIVIDER)

    errors, pcts = [], []
    rows = []

    for cpu in BENCHMARK_CPUS:
        label = cpu["name"]
        try:
            row = build_row(cpu, all_cols, fam_cols, tier_cols)
        except Exception as exc:
            print(f"  [FEATURE BUILD ERROR] {label}")
            print(f"      {exc}")
            rows.append((label, None, None, None, None, "feature_error"))
            continue

        nan_cols = row.columns[row.isna().any()].tolist()
        if nan_cols:
            # Still predict; E-only CPUs fall back to Cores for ratio features
            nan_note = f"NaN in: {nan_cols}"
        else:
            nan_note = None

        try:
            pred = model.predict(poly.transform(scaler.transform(row.fillna(0))))[0]
        except Exception as exc:
            print(f"  [PREDICT ERROR] {label}")
            print(f"      {exc}")
            rows.append((label, cpu.get("actual_ghz"), None, None, None, "predict_error"))
            continue

        actual = cpu.get("actual_ghz")
        if actual is not None:
            err = pred - actual
            pct = abs(err) / actual * 100
            errors.append(abs(err))
            pcts.append(pct)
            flag = " ← LARGE ERROR" if abs(err) > 0.5 else ""
            print(
                f"  {label:<45} {actual:>6.2f}G {pred:>6.2f}G "
                f"{err:>+6.3f} {pct:>5.1f}%{flag}"
            )
            rows.append((label, actual, pred, err, pct, nan_note or "ok"))
        else:
            print(f"  {label:<45} {'N/A':>7} {pred:>6.2f}G {'—':>7} {'—':>6}")
            rows.append((label, None, pred, None, None, nan_note or "ok"))

        if nan_note:
            print(f"    ⚠  {nan_note}")

    # Summary
    print(DIVIDER)
    if errors:
        print(
            f"  MAE across {len(errors)} CPUs with known turbo: "
            f"{np.mean(errors):.3f} GHz  |  "
            f"Max abs error: {np.max(errors):.3f} GHz"
        )
        print(f"  Mean |%error| : {np.mean(pcts):.1f}%")

    print(f"{'═' * 72}\n")

    # Flagged issues
    issues = [(r[0], r[5]) for r in rows if r[5] not in ("ok", None)]
    if issues:
        print("ISSUES DETECTED:")
        for name, note in issues:
            print(f"  • {name}")
            print(f"      {note}")
        print()

    return rows


# =========================================================================
# 7.  ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Ridge model + run CPU benchmark suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples
        --------
          python predict_bench.py
          python predict_bench.py --data /home/user/intel_data
        """),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to directory containing the Intel CSV files. "
             "Defaults to ../data/ relative to this script.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data) if args.data else Path(__file__).resolve().parent.parent / "data"
    print(f"\nData directory : {data_dir}")

    print("\n[1/3] Loading & engineering features …")
    try:
        df = load_and_engineer(data_dir)
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    print("\n[2/3] Training Ridge model …")
    model, poly, scaler, all_cols, fam_cols, tier_cols = build_model(df)

    print("\n[3/3] Running benchmark suite …")
    run_benchmarks(model, poly, scaler, all_cols, fam_cols, tier_cols)


if __name__ == "__main__":
    main()