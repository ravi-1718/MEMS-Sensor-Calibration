
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sensor_calibration.csv"
MODEL_DIR = PROJECT_DIR / "models"

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

TARGETS = [
    f"tco_{axis}_true_mg_per_C" for axis in "xyz"
] + [
    f"tcs_{axis}_true_ppm_per_C" for axis in "xyz"
] + [
    f"offset_{axis}_85C_mg" for axis in "xyz"
]


def build_pipeline():
    categorical = ["derivative", "package"]
    numeric = [col for col in FEATURES if col not in categorical]

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    df = pd.read_csv(DATA_PATH)

    required = FEATURES + TARGETS + ["characterized_3temp"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Use only characterized records for supervised training.
    training_data = df.loc[df["characterized_3temp"].eq(True)].copy()

    if training_data[TARGETS].isna().any().any():
        raise ValueError("Training targets contain missing values.")

    pipeline = build_pipeline()
    pipeline.fit(training_data[FEATURES], training_data[TARGETS])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "deployment_pipeline.joblib"

    joblib.dump({
        "pipeline": pipeline,
        "features": FEATURES,
        "targets": TARGETS,
        "model_name": "Extra Trees Regressor",
        "training_rows": len(training_data),
        "training_wafers": int(training_data["wafer_id"].nunique()),
    }, model_path)

    print("Deployment model created successfully.")
    print(f"Training rows: {len(training_data)}")
    print(f"Training wafers: {training_data['wafer_id'].nunique()}")
    print(f"Input features: {len(FEATURES)}")
    print(f"Prediction targets: {len(TARGETS)}")
    print(f"Saved to: {model_path}")
    print("\nFinal test metrics remain those from evaluate_final.py.")


if __name__ == "__main__":
    main()
