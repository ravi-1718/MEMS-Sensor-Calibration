
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "deployment_pipeline.joblib"
METRICS_PATH = (
    PROJECT_DIR / "reports" / "extra_trees_final_test_metrics.csv"
)

st.set_page_config(
    page_title="MEMS Sensor Calibration AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# 2. Load the trained deployment model
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
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# --------------------------------------------------
# 3. Application header
# --------------------------------------------------

st.title("AI-Based MEMS Sensor Calibration Prediction")

st.write(
    "An ML-based demonstration system for predicting "
    "temperature-related sensor calibration parameters."
)

st.warning(
    "This application uses a synthetic dataset. "
    "Predictions have not been validated for production hardware."
)

# --------------------------------------------------
# 4. Navigation tabs
# --------------------------------------------------

home_tab, prediction_tab, performance_tab = st.tabs(
    [
        "🏠 Home",
        "🔬 Make Predictions",
        "📈 Model Performance",
    ]
)

# --------------------------------------------------
# TAB 1: HOME
# --------------------------------------------------

with home_tab:
    st.header("Project Overview")

    st.write(
        "This project uses a trained Extra Trees regression "
        "pipeline to estimate MEMS sensor calibration parameters "
        "from manufacturing and room-temperature measurements."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input Features", len(features))

    with col2:
        st.metric("Prediction Targets", len(targets))

    with col3:
        st.metric("Training Records", f"{bundle['training_rows']:,}")

    st.divider()

    st.subheader("What can this application do?")

    st.markdown(
        """
        - Upload a CSV file containing sensor input measurements.
        - Validate required input columns.
        - Predict TCO for the X, Y and Z axes.
        - Predict TCS for the X, Y and Z axes.
        - Predict sensor offset at 85°C for the X, Y and Z axes.
        - Visualize predicted values using a line chart.
        - Download predictions as a CSV file.
        - Review the model's previously measured final-test metrics.
        """
    )

    st.subheader("Prediction targets")

    target_info = pd.DataFrame(
        {
            "Parameter": [
                "TCO X", "TCO Y", "TCO Z",
                "TCS X", "TCS Y", "TCS Z",
                "Offset X at 85°C",
                "Offset Y at 85°C",
                "Offset Z at 85°C",
            ],
            "Target column": targets,
            "Meaning": [
                "Temperature coefficient of offset, X axis",
                "Temperature coefficient of offset, Y axis",
                "Temperature coefficient of offset, Z axis",
                "Temperature coefficient of sensitivity, X axis",
                "Temperature coefficient of sensitivity, Y axis",
                "Temperature coefficient of sensitivity, Z axis",
                "Predicted sensor offset at 85°C, X axis",
                "Predicted sensor offset at 85°C, Y axis",
                "Predicted sensor offset at 85°C, Z axis",
            ],
        }
    )

    st.dataframe(
        target_info,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "To generate predictions, open the Make Predictions tab "
        "and upload a CSV containing all required input features."
    )

    st.caption(
        "The model is a demonstration using synthetic data. "
        "Its results do not establish performance on real hardware."
    )


# --------------------------------------------------
# TAB 2: MAKE PREDICTIONS
# --------------------------------------------------

with prediction_tab:
    st.header("Generate Sensor Calibration Predictions")

    st.write(
        "Upload a CSV file with the required input features. "
        "Existing target columns in the uploaded file are not "
        "used as model inputs."
    )

    uploaded_file = st.file_uploader(
        "Choose a sensor CSV file",
        type=["csv"],
        key="sensor_csv_upload",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
    else:
        try:
            input_df = pd.read_csv(uploaded_file)

            if input_df.empty:
                st.error("The uploaded CSV contains no rows.")
                st.stop()

            st.write(f"Uploaded rows: {len(input_df):,}")
            st.write(f"Uploaded columns: {len(input_df.columns)}")

            missing_features = [
                column
                for column in features
                if column not in input_df.columns
            ]

            if missing_features:
                st.error(
                    "Missing required input columns: "
                    + ", ".join(missing_features)
                )
                st.stop()

            # Only approved model input features are passed
            # to the prediction pipeline.
            X = input_df[features].copy()

            # Reject columns that have no usable values.
            all_missing = X.columns[X.isna().all()].tolist()

            if all_missing:
                st.error(
                    "These required features are entirely empty: "
                    + ", ".join(all_missing)
                )
                st.stop()

            st.info(
                "Individual missing values may be imputed by "
                "the trained pipeline. Entirely empty feature "
                "columns are rejected."
            )

            # Display characterization count when the flag exists.
            if "characterized_3temp" in input_df.columns:
                characterized_count = int(
                    input_df["characterized_3temp"]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )

                st.info(
                    f"Rows marked as characterized_3temp: "
                    f"{characterized_count:,} of {len(input_df):,}. "
                    "This flag is informational and is not a model input."
                )

            if st.button(
                "Generate predictions",
                type="primary",
                key="generate_predictions",
            ):
                with st.spinner("Generating predictions..."):
                    prediction_values = pipeline.predict(X)

                predictions_df = pd.DataFrame(
                    prediction_values,
                    columns=targets,
                    index=input_df.index,
                )

                # Store results so users can change chart targets
                # without having to upload the file again.
                st.session_state["mems_input_df"] = input_df.copy()
                st.session_state["mems_predictions_df"] = (
                    predictions_df.copy()
                )

                st.success("Predictions generated successfully.")

            predictions_df = st.session_state.get(
                "mems_predictions_df"
            )

            saved_input_df = st.session_state.get(
                "mems_input_df"
            )

            if predictions_df is not None and saved_input_df is not None:
                st.divider()

                st.subheader("Prediction Results")

                st.dataframe(
                    predictions_df,
                    width="stretch",
                    height=350,
                )

                st.write(
                    f"Total predicted records: "
                    f"{len(predictions_df):,}"
                )

                st.subheader("Visualize Predictions")

                selected_target = st.selectbox(
                    "Choose a parameter to visualize",
                    options=targets,
                    key="prediction_chart_target",
                )

                chart_df = predictions_df[[selected_target]].reset_index(
                    drop=True
                )

                chart_df.index.name = "Record"

                st.line_chart(chart_df)

                st.subheader("Download Results")

                output_df = pd.concat(
                    [
                        saved_input_df.reset_index(drop=True),
                        predictions_df.add_prefix(
                            "predicted_"
                        ).reset_index(drop=True),
                    ],
                    axis=1,
                )

                csv_bytes = output_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_bytes,
                    file_name="sensor_calibration_predictions.csv",
                    mime="text/csv",
                    key="download_predictions",
                )

                st.caption(
                    "Predictions are estimates. Validate them against "
                    "real hardware and engineering tolerances before use."
                )

        except Exception as exc:
            st.error(f"Could not process this file: {exc}")


# --------------------------------------------------
# TAB 3: MODEL PERFORMANCE
# --------------------------------------------------

with performance_tab:
    st.header("Final Model Performance")

    st.write(
        "These metrics come from the previously completed final "
        "test evaluation on 455 characterized records from four "
        "unseen wafers. The test set was not used to retrain or "
        "tune the model."
    )

    if not METRICS_PATH.exists():
        st.error(
            "Final test metrics file not found: "
            f"{METRICS_PATH}"
        )
    else:
        try:
            metrics_df = pd.read_csv(METRICS_PATH)

            required_metric_columns = {
                "target", "MAE", "RMSE", "R2"
            }

            if not required_metric_columns.issubset(
                metrics_df.columns
            ):
                st.error(
                    "The metrics CSV does not contain the "
                    "expected columns: target, MAE, RMSE, R2."
                )
            else:
                display_df = metrics_df.copy()

                for column in ["MAE", "RMSE", "R2"]:
                    display_df[column] = pd.to_numeric(
                        display_df[column],
                        errors="coerce",
                    )

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "MAE and RMSE: lower is generally better. "
                    "R² closer to 1 generally indicates better "
                    "explanatory performance. Interpret all metrics "
                    "in the context of each target's units and range."
                )

                metrics_csv = metrics_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download final test metrics",
                    data=metrics_csv,
                    file_name="extra_trees_final_test_metrics.csv",
                    mime="text/csv",
                    key="download_metrics",
                )

        except Exception as exc:
            st.error(f"Could not load model metrics: {exc}")

    st.divider()

    st.subheader("Evaluation notes")

    st.markdown(
        """
        - The final test set contains 455 characterized records.
        - The records belong to four wafers excluded from model training.
        - The model was evaluated on wafers not seen during training.
        - The reported metrics are from the completed final evaluation.
        - The deployment model was subsequently trained on all 2,174
          characterized synthetic records from 20 wafers.
        """
    )

    st.warning(
        "All reported metrics are based on a synthetic dataset. "
        "They do not establish accuracy, safety, or production "
        "readiness on real MEMS sensors."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based MEMS Sensor Calibration Prediction and Optimization "
    "System | Research demonstration using synthetic data"
)
