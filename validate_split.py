
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sensor_calibration.csv"

TARGETS = [
    f"tco_{axis}_true_mg_per_C" for axis in "xyz"
] + [
    f"tcs_{axis}_true_ppm_per_C" for axis in "xyz"
] + [
    f"offset_{axis}_85C_mg" for axis in "xyz"
]

FEATURES = [
    "derivative", "full_scale_g", "nominal_sens_LSB_per_g",
    "package",
    "die_x", "die_y", "x_mm", "y_mm", "radius_mm",
    "pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq",
    "pcm_tox_A", "pcm_mobility_rel",
    "stress_gauge_mV", "asic_vref_mV",
    "mems_resonance_kHz", "mems_q_factor",
    "st_response_x_mg", "st_response_y_mg", "st_response_z_mg",
] + [
    f"offset_{axis}_25C_mg" for axis in "xyz"
] + [
    f"sens_{axis}_25C_rel" for axis in "xyz"
]


def main():
    df = pd.read_csv(DATA_PATH)

    # Supervised training/evaluation uses fully characterized records only.
    labelled = df.loc[df["characterized_3temp"].eq(True)].copy()

    required = FEATURES + TARGETS + ["wafer_id"]
    missing = sorted(set(required) - set(labelled.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if labelled[required].isna().any().any():
        raise ValueError("Missing values detected in required columns.")

    groups = labelled["wafer_id"]

    # First reserve 20% of wafers for the final test set.
    outer_split = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val_idx, test_idx = next(
        outer_split.split(labelled, groups=groups)
    )

    train_val = labelled.iloc[train_val_idx].copy()
    test = labelled.iloc[test_idx].copy()

    # From the remaining 80%, reserve 25% for validation.
    # This is 20% of the full set.
    inner_split = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=43
    )
    train_idx, val_idx = next(
        inner_split.split(train_val, groups=train_val["wafer_id"])
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()

    partitions = {
        "TRAIN": train,
        "VALIDATION": validation,
        "FINAL TEST": test,
    }

    print("=" * 60)
    print("WAFER-GROUPED SPLIT VALIDATION")
    print("=" * 60)

    for name, part in partitions.items():
        print(f"\n{name}")
        print(f"  Rows: {len(part)}")
        print(f"  Wafers: {part['wafer_id'].nunique()}")
        print(f"  Wafer IDs: {sorted(part['wafer_id'].unique())}")

    # Confirm that no wafer appears in more than one partition.
    names = list(partitions)
    valid = True

    print("\n--- Leakage checks ---")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = partitions[names[i]], partitions[names[j]]
            overlap = set(left["wafer_id"]) & set(right["wafer_id"])
            print(f"{names[i]} vs {names[j]} wafer overlap: {sorted(overlap)}")
            if overlap:
                valid = False

    print(f"\nFeature count: {len(FEATURES)}")
    print(f"Target count: {len(TARGETS)}")
    print(f"Feature/target overlap: {set(FEATURES) & set(TARGETS)}")

    if not valid:
        raise RuntimeError("FAILED: wafer overlap detected.")

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise RuntimeError("FAILED: one or more partitions are empty.")

    print("\nPASS: no wafer overlaps between partitions.")
    print("No model was trained. No files were modified.")


if __name__ == "__main__":
    main()
