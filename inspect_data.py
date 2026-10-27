
from pathlib import Path

import pandas as pd

# Locate the dataset relative to this script.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sensor_calibration.csv"

print("=" * 60)
print("MEMS SENSOR CALIBRATION - DATA INSPECTION")
print("=" * 60)

if not DATA_PATH.exists():
    print(f"\nERROR: Dataset not found at:\n{DATA_PATH}")
    print("\nCheck that sensor_calibration.csv is inside the data folder.")
    raise SystemExit(1)

# Load the CSV without changing the original file.
df = pd.read_csv(DATA_PATH)

print(f"\nDataset path: {DATA_PATH}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

print("\n--- First 5 rows ---")
print(df.head().to_string())

print("\n--- Column names and data types ---")
print(df.dtypes.to_string())

print("\n--- Missing values (columns with missing data) ---")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values detected.")
else:
    print(missing.to_string())

print("\n--- Duplicate rows ---")
print(f"Exact duplicate rows: {df.duplicated().sum()}")

print("\n--- Important dataset columns ---")
for col in ["die_id", "wafer_id", "lot_id", "characterized_3temp"]:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique(dropna=True)}")
        if col == "characterized_3temp":
            print(df[col].value_counts(dropna=False).to_string())
    else:
        print(f"\n{col}: NOT FOUND")

print("\n--- Candidate prediction targets ---")
target_prefixes = ("tco_", "tcs_", "offset_")
target_cols = [
    col for col in df.columns
    if col.startswith(target_prefixes)
    and (
        col.endswith("_true_mg_per_C")
        or col.endswith("_true_ppm_per_C")
        or "_85C_" in col
    )
]

if target_cols:
    for col in target_cols:
        print(f"{col}: {df[col].notna().sum():,} non-missing values")
else:
    print("No candidate targets matched. We will inspect the actual schema.")

print("\n--- Wafer counts ---")
if "wafer_id" in df.columns:
    print(f"Unique wafers: {df['wafer_id'].nunique(dropna=True)}")
    print("Rows per wafer (summary):")
    print(df.groupby("wafer_id", dropna=False).size().describe().to_string())

print("\nInspection complete. The original CSV was not modified.")
