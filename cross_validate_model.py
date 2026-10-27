
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
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

    # Recreate the same 12-wafer training partition.
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val_idx, _ = next(
        outer.split(data, groups=data["wafer_id"])
    )

    train_val = data.iloc[train_val_idx].copy()

    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=43
    )
    train_idx, _ = next(
        inner.split(train_val, groups=train_val["wafer_id"])
    )
    training = train_val.iloc[train_idx].copy()

    X = training[FEATURES]
    y = training[TARGETS]
    groups = training["wafer_id"]

    n_splits = min(4, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    rows = []

    for fold, (fit_idx, val_idx) in enumerate(
        cv.split(X, y, groups=groups), start=1
    ):
        model = build_pipeline()
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        predictions = model.predict(X.iloc[val_idx])

        actual = y.iloc[val_idx]

        assert set(groups.iloc[fit_idx]).isdisjoint(
            set(groups.iloc[val_idx])
        )

        for j, target in enumerate(TARGETS):
            rows.append({
                "fold": fold,
                "target": target,
                "MAE": mean_absolute_error(
                    actual.iloc[:, j], predictions[:, j]
                ),
                "RMSE": mean_squared_error(
                    actual.iloc[:, j], predictions[:, j]
                ) ** 0.5,
                "R2": r2_score(
                    actual.iloc[:, j], predictions[:, j]
                ),
            })

        print(
            f"Fold {fold}: "
            f"{len(fit_idx)} fit rows, "
            f"{len(val_idx)} validation rows; "
            "wafer overlap = 0"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    output = REPORT_DIR / "extra_trees_group_cv.csv"
    results.to_csv(output, index=False)

    print("\n--- Cross-validation summary ---")
    summary = results.groupby("target")[["MAE", "RMSE", "R2"]].agg(
        ["mean", "std"]
    )
    print(summary.round(4).to_string())
    print(f"\nSaved results to: {output}")
    print("The final test wafers were not used.")


if __name__ == "__main__":
    main()
