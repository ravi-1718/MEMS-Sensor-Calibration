
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "deployment_pipeline.joblib"
METRICS_PATH = PROJECT_DIR / "reports" / "extra_trees_final_test_metrics.csv"

st.set_page_config(
    page_title="MEMS Sensor Calibration AI",
    page_icon="📊",
    layout="wide",
)

st.title("AI-Based MEMS Sensor Calibration Prediction")
st.write(
    "Predict TCO, TCS and 85°C sensor offset using eligible "
    "manufacturing and room-temperature measurements."
)

st.warning(
    "This application uses a synthetic dataset for demonstration. "
    "Predictions are not validated for production hardware."
)

# --------------------------------------------------
# Load the trained deployment model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python build_deployment_model.py first."
        )
    return joblib.load(MODEL_PATH)


try:
    bundle = load_model()
    pipeline = bundle["pipeline"]
    features = bundle["features"]
    targets = bundle["targets"]
except Exception as exc:
    st.error(f"Unable to load the model: {exc}")
    st.stop()


# --------------------------------------------------
# Input data upload and validation
# --------------------------------------------------

st.subheader("1. Upload input data")
st.write(
    "Upload a CSV containing all required input features. "
    "Target columns are not used as model inputs."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        if input_df.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        st.write(f"Uploaded rows: {len(input_df):,}")
        st.write(f"Uploaded columns: {len(input_df.columns)}")

        missing_features = [
            col for col in features if col not in input_df.columns
        ]

        if missing_features:
            st.error(
                "Missing required input columns: "
                + ", ".join(missing_features)
            )
            st.stop()

        # Only approved input features enter the model.
        X = input_df[features].copy()

        # Reject columns that contain no usable values.
        all_missing = X.columns[X.isna().all()].tolist()

        if all_missing:
            st.error(
                "These required features contain no usable values: "
                + ", ".join(all_missing)
            )
            st.stop()

        # Check for non-numeric values in numeric features.
        numeric_features = X.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        # The model pipeline can impute missing values, but it cannot
        # safely interpret arbitrary text in numeric measurement fields.
        invalid_numeric = []

        for col in numeric_features:
            converted = pd.to_numeric(X[col], errors="coerce")
            invalid_count = int(
                (converted.isna() & X[col].notna()).sum()
            )

            if invalid_count:
                invalid_numeric.append(
                    f"{col} ({invalid_count} invalid values)"
                )
            else:
                X[col] = converted

        if invalid_numeric:
            st.error(
                "Some numeric input columns contain invalid values: "
                + ", ".join(invalid_numeric)
            )
            st.stop()

        st.info(
            "Missing individual feature values may be imputed by the "
            "trained pipeline. Entirely empty feature columns are rejected."
        )

        if "characterized_3temp" in input_df.columns:
            characterized_count = int(
                (
                    input_df["characterized_3temp"]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(["true", "1", "yes"])
                ).sum()
            )

            st.info(
                f"Rows marked as characterized_3temp: "
                f"{characterized_count:,} of {len(input_df):,}. "
                "This flag is informational and is not a model input."
            )

        generate = st.button(
            "Generate predictions",
            type="primary",
        )

        if generate:
            with st.spinner("Generating predictions..."):
                prediction_values = pipeline.predict(X)

            predictions_df = pd.DataFrame(
                prediction_values,
                columns=targets,
                index=input_df.index,
            )

            st.success(
                f"Predictions generated for {len(predictions_df):,} rows."
            )

            # ------------------------------------------
            # Prediction results
            # ------------------------------------------

            st.subheader("2. Prediction results")

            st.caption(
                "These are model estimates, not certified calibration "
                "values. The model was trained on synthetic data."
            )

            st.dataframe(
                predictions_df,
                width="stretch",
            )

            # ------------------------------------------
            # Visualize predictions
            # ------------------------------------------

            st.subheader("3. Visualize predictions")

            selected_target = st.selectbox(
                "Choose a target",
                options=targets,
            )

            st.line_chart(
                predictions_df[selected_target],
                width="stretch",
            )

            # ------------------------------------------
            # Download predictions
            # ------------------------------------------

            st.subheader("4. Download results")

            output_df = pd.concat(
                [
                    input_df.reset_index(drop=True),
                    predictions_df.add_prefix("predicted_").reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            )

            csv_bytes = output_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name="sensor_calibration_predictions.csv",
                mime="text/csv",
            )

            st.caption(
                "Validate predictions against real hardware and "
                "engineering tolerances before any practical use."
            )

    except Exception as exc:
        st.error(f"Could not process this file: {exc}")

else:
    st.info("Upload a CSV file to begin.")


# --------------------------------------------------
# Final model performance report
# --------------------------------------------------

st.divider()
st.subheader("5. Final Model Performance")

st.write(
    "The following metrics come from the previously completed final "
    "test evaluation on 455 characterized records from four unseen wafers. "
    "These results are reported as measured; the test set is not used "
    "to retrain or tune the model."
)

if METRICS_PATH.exists():
    try:
        metrics_df = pd.read_csv(METRICS_PATH)

        required_metric_columns = {"target", "MAE", "RMSE", "R2"}

        if not required_metric_columns.issubset(metrics_df.columns):
            st.error(
                "The metrics file does not contain the expected columns: "
                "target, MAE, RMSE and R2."
            )
        else:
            st.dataframe(
                metrics_df,
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "MAE and RMSE: lower is generally better. "
                "R²: closer to 1 indicates better explanatory performance. "
                "Metrics are based on a synthetic dataset and do not "
                "establish performance on real MEMS hardware."
            )

            metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download final test metrics",
                data=metrics_csv,
                file_name="extra_trees_final_test_metrics.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(f"Could not load final test metrics: {exc}")
else:
    st.warning(
        "Final test metrics file was not found. Expected location: "
        f"{METRICS_PATH}"
    )
