
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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


def create_splits(df):
    data = df.loc[df["characterized_3temp"].eq(True)].copy()

    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val_idx, test_idx = next(
        outer.split(data, groups=data["wafer_id"])
    )

    train_val = data.iloc[train_val_idx].copy()
    test = data.iloc[test_idx].copy()

    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=43
    )
    train_idx, val_idx = next(
        inner.split(train_val, groups=train_val["wafer_id"])
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[val_idx].copy()

    assert set(train.wafer_id).isdisjoint(validation.wafer_id)
    assert set(train.wafer_id).isdisjoint(test.wafer_id)
    assert set(validation.wafer_id).isdisjoint(test.wafer_id)

    return train, validation, test


def build_pipeline(model):
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
        ("model", model),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    train, validation, test = create_splits(df)

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, estimator in candidates.items():
        print(f"\nTraining {name}...")
        pipeline = build_pipeline(estimator)
        pipeline.fit(train[FEATURES], train[TARGETS])
        predictions = pipeline.predict(validation[FEATURES])

        for i, target in enumerate(TARGETS):
            actual = validation[target].to_numpy()
            predicted = predictions[:, i]

            results.append({
                "model": name,
                "target": target,
                "MAE": mean_absolute_error(actual, predicted),
                "RMSE": mean_squared_error(actual, predicted) ** 0.5,
                "R2": r2_score(actual, predicted),
            })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "model_comparison_validation.csv"

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON - VALIDATION ONLY")
    print("=" * 75)
    print(
        results_df.sort_values(["target", "MAE"])
        .round(5)
        .to_string(index=False)
    )

    print(f"\nSaved comparison to: {output_path}")
    print(f"Training wafers: {train['wafer_id'].nunique()}")
    print(f"Validation wafers: {validation['wafer_id'].nunique()}")
    print(f"Reserved final test wafers: {test['wafer_id'].nunique()}")
    print("\nThe final test set has not been evaluated.")


if __name__ == "__main__":
    main()
