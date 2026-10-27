
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sensor_calibration.csv"
REPORT_DIR = PROJECT_DIR / "reports"

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


def build_pipeline():
    categorical = ["derivative", "package"]
    numeric = [c for c in FEATURES if c not in categorical]

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
    data = df.loc[df["characterized_3temp"].eq(True)].copy()

    # Recreate the original reserved test wafers.
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        outer.split(data, groups=data["wafer_id"])
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    # Guard against accidental group leakage.
    train_wafers = set(train["wafer_id"])
    test_wafers = set(test["wafer_id"])

    if not train_wafers.isdisjoint(test_wafers):
        raise RuntimeError("Leakage detected: train/test wafers overlap.")

    print("Training wafers:", len(train_wafers))
    print("Test wafers:", len(test_wafers))
    print("Wafer overlap:", train_wafers & test_wafers)
    print("Training rows:", len(train))
    print("Test rows:", len(test))

    # Fit once on all non-test wafers.
    model = build_pipeline()
    model.fit(train[FEATURES], train[TARGETS])

    # First and only final-test evaluation in this script.
    predictions = model.predict(test[FEATURES])

    metric_rows = []
    prediction_report = test[
        ["die_id", "lot_id", "wafer_id"] + TARGETS
    ].copy()

    for i, target in enumerate(TARGETS):
        actual = test[target].to_numpy()
        predicted = predictions[:, i]

        metric_rows.append({
            "target": target,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted),
        })

        prediction_report[f"predicted_{target}"] = predicted
        prediction_report[f"absolute_error_{target}"] = abs(
            actual - predicted
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = REPORT_DIR / "extra_trees_final_test_metrics.csv"
    predictions_path = REPORT_DIR / "extra_trees_final_test_predictions.csv"

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    prediction_report.to_csv(predictions_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS - EXTRA TREES")
    print("=" * 65)
    print(pd.DataFrame(metric_rows).round(5).to_string(index=False))
    print("\nSaved metrics:", metrics_path)
    print("Saved predictions:", predictions_path)
    print("\nDo not use these results to tune the model.")


if __name__ == "__main__":
    main()
