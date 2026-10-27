
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


def make_splits(df):
    from sklearn.model_selection import GroupShuffleSplit

    data = df.loc[df["characterized_3temp"].eq(True)].copy()
    groups = data["wafer_id"]

    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_val_idx, test_idx = next(outer.split(data, groups=groups))

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

    # Confirm the expected split has not accidentally changed.
    sets = [
        set(train["wafer_id"]),
        set(validation["wafer_id"]),
        set(test["wafer_id"]),
    ]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])

    return train, validation, test


def main():
    df = pd.read_csv(DATA_PATH)
    train, validation, test = make_splits(df)

    # Fit the baseline on training labels only.
    model = DummyRegressor(strategy="mean")
    model.fit(train[FEATURES], train[TARGETS])

    # Validation data is used for the first performance comparison.
    predictions = model.predict(validation[FEATURES])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, target in enumerate(TARGETS):
        actual = validation[target].to_numpy()
        predicted = predictions[:, i]

        results.append({
            "target": target,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": mean_squared_error(actual, predicted) ** 0.5,
            "R2": r2_score(actual, predicted),
        })

    results_df = pd.DataFrame(results)
    output_path = REPORT_DIR / "baseline_validation_metrics.csv"
    results_df.to_csv(output_path, index=False)

    print("=" * 60)
    print("BASELINE MODEL - VALIDATION RESULTS")
    print("=" * 60)
    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print("\nMetrics by target:")
    print(results_df.round(5).to_string(index=False))
    print(f"\nSaved results to: {output_path}")
    print("\nNote: the final test set has not been evaluated.")


if __name__ == "__main__":
    main()
