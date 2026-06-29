

import pandas as pd
from pathlib import Path

data_dir = Path("data")

# All 7 CSV files that share the 16-column schema
csv_files = [
    data_dir / "Core-Processors-1.8.csv",
    data_dir / "Xeon-Processors-1.8.csv",
    data_dir / "Atom-Processors-1.9-16col.csv",
    data_dir / "Celeron-Processors-1.9-16col.csv",
    data_dir / "Core-Ultra-Processors-1.10-16col.csv",
    data_dir / "Intel-Processors-1.9-16col.csv",
    data_dir / "Pentium-Processors-1.9-16col.csv",
]

# Read each file and collect the DataFrames
frames = []
for f in csv_files:
    print(f"Loading {f.name} ...")
    df = pd.read_csv(f)
    frames.append(df)

full = pd.concat(frames, ignore_index=True)
print(full.dtypes)

# Our basic model will use Lithography, Cores, and TDP to predict Max. Turbo Freq

