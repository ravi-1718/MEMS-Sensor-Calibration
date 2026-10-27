
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sensor_calibration.csv"

df = pd.read_csv(DATA_PATH)

targets = [
    f"tco_{axis}_true_mg_per_C" for axis in "xyz"
] + [
    f"tcs_{axis}_true_ppm_per_C" for axis in "xyz"
] + [
    f"offset_{axis}_85C_mg" for axis in "xyz"
]

allowed_features = [
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

print("=" * 60)
print("LEAKAGE AUDIT - BEFORE MODEL TRAINING")
print("=" * 60)

missing_features = [c for c in allowed_features if c not in df.columns]
missing_targets = [c for c in targets if c not in df.columns]

print("\n--- Feature allowlist ---")
print(f"Expected features: {len(allowed_features)}")
print(f"Missing features: {missing_features or 'None'}")

print("\n--- Target verification ---")
print(f"Expected targets: {len(targets)}")
print(f"Missing targets: {missing_targets or 'None'}")

if "characterized_3temp" not in df.columns:
    raise ValueError("characterized_3temp is missing.")

if "wafer_id" not in df.columns:
    raise ValueError("wafer_id is missing.")

# Check that targets are complete in the proposed labelled population.
labelled = df.loc[df["characterized_3temp"] == True].copy()

print("\n--- Labelled population ---")
print(f"Rows marked characterized: {len(labelled)}")
print(f"Unique labelled wafers: {labelled['wafer_id'].nunique()}")

if not missing_targets:
    print("\nMissing target values in characterized rows:")
    print(labelled[targets].isna().sum().to_string())

print("\n--- Characterized rows per wafer ---")
print(
    df.groupby("wafer_id")["characterized_3temp"]
    .agg(total_rows="size", characterized_rows="sum")
    .to_string()
)

# Explicitly prohibited model-input columns.
forbidden = [
    c for c in df.columns
    if c in targets
    or c == "latent_defect"
    or c in ["die_id", "lot_id", "wafer_id", "characterized_3temp"]
    or "_m40C_" in c
    or "_85C_" in c
    or "post_reflow" in c
    or c.startswith("cal_")
    or c.startswith("tco")
    or c.startswith("tcs")
    or c.startswith("tco2")
    or c.startswith("conv_test_")
]

print("\n--- Forbidden columns (never use as features initially) ---")
print("\n".join(forbidden))

print("\n--- Feature/target overlap check ---")
overlap = sorted(set(allowed_features) & set(targets))
print(f"Overlap: {overlap or 'None'}")

if missing_features or missing_targets or overlap:
    print("\nAUDIT STATUS: ACTION REQUIRED. Resolve the issues above.")
else:
    print("\nAUDIT STATUS: BASIC CHECKS PASSED.")
    print("This is not yet proof that all features are available at prediction time.")

print("\nAudit complete. No model was trained and no CSV was changed.")
