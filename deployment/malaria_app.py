import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Prediction App",
    page_icon="🦟",
    layout="wide"
)

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")

model_path = os.path.join(MODEL_DIR, "malaria_model.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
features_path = os.path.join(MODEL_DIR, "feature_names.pkl")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("🦟 Malaria Prediction App")
st.sidebar.markdown("""
Predict malaria cases in Nigerian LGAs.

**Links:**
- [GitHub Repository](https://github.com/SamuelComputer/malaria-prediction-system/tree/main)
- [Sunday Iyanu Samuel](https://www.linkedin.com/in/sunday-iyanu-samuel)
""")

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("🦟 Malaria Cases Prediction in Nigerian LGAs")

st.markdown("""
Predict total malaria cases in Nigerian Local Government Areas (LGAs) based on
environmental, geographic, seasonal, and intervention factors.
""")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_files():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    return model, scaler, features

try:
    model, scaler, all_features = load_files()
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# ─────────────────────────────────────────────
# GET STATES & LGAs
# ─────────────────────────────────────────────
states = sorted(list(set([
    f.replace("state_", "")
    for f in all_features
    if f.startswith("state_")
])))

lgas = sorted(list(set([
    f.replace("lga_", "")
    for f in all_features
    if f.startswith("lga_")
])))

# ─────────────────────────────────────────────
# MONTHS
# ─────────────────────────────────────────────
month_mapping = {
    "January": 1, "February": 2, "March": 3,
    "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9,
    "October": 10, "November": 11, "December": 12
}

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.header("📊 Input Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Geographic & Temporal")

    selected_state = st.selectbox("Select State", states)

    state_lgas = [lga for lga in lgas if lga.startswith(selected_state)]
    selected_lga = st.selectbox("Select LGA", state_lgas)

    st.info(f"📍 {selected_state} → {selected_lga}")

    year = st.number_input("Year", min_value=2018, max_value=2030, value=2023)

    selected_month_name = st.selectbox("Month", list(month_mapping.keys()), index=5)
    month = month_mapping[selected_month_name]

    season = st.selectbox("Season", ["Dry", "Rainy"])
    intervention = st.selectbox("Intervention Applied?", ["No", "Yes"])

with col2:
    st.subheader("🌧️ Environmental & Data Quality")

    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=600.0, value=200.0, step=10.0)
    temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=40.0, value=27.0, step=0.5)
    vegetation = st.number_input("Vegetation Index", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    reporting = st.slider("Reporting Completeness (%)", 0, 100, 80)
    timeliness = st.slider("Timeliness (%)", 0, 100, 80)

# ─────────────────────────────────────────────
# PREDICTION BUTTON
# ─────────────────────────────────────────────
st.markdown("---")
predict_button = st.button("🔮 Predict Malaria Cases", type="primary")

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_button:

    try:
        input_data = pd.DataFrame(
            np.zeros((1, len(all_features))),
            columns=all_features
        )

        if "year" in input_data.columns:
            input_data["year"] = year
        if "month" in input_data.columns:
            input_data["month"] = month
        if "rainfall_mm" in input_data.columns:
            input_data["rainfall_mm"] = rainfall
        if "temperature_avg_c" in input_data.columns:
            input_data["temperature_avg_c"] = temperature
        if "vegetation_index" in input_data.columns:
            input_data["vegetation_index"] = vegetation
        if "reporting_completeness_pct" in input_data.columns:
            input_data["reporting_completeness_pct"] = reporting
        if "timeliness_pct" in input_data.columns:
            input_data["timeliness_pct"] = timeliness
        if "intervention_flag" in input_data.columns:
            input_data["intervention_flag"] = 1 if intervention == "Yes" else 0
        if "season_Rainy" in input_data.columns:
            input_data["season_Rainy"] = 1 if season == "Rainy" else 0

        state_col = f"state_{selected_state}"
        lga_col = f"lga_{selected_lga}"

        if state_col in input_data.columns:
            input_data[state_col] = 1
        if lga_col in input_data.columns:
            input_data[lga_col] = 1

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # ─────────────────────────────────────────
        # RESULTS
        # ─────────────────────────────────────────
        st.markdown("---")
        st.success("✅ Prediction Complete!")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Predicted Total Cases", f"{prediction:,.0f}")
        with metric_col2:
            st.metric("State", selected_state)
        with metric_col3:
            st.metric("LGA", selected_lga)

        # ─────────────────────────────────────────
        # RISK ASSESSMENT
        # ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Risk Assessment")

        if prediction < 2500:
            risk = "🟢 LOW RISK"
            recommendation = "Standard monitoring protocols sufficient"
        elif prediction < 11000:
            risk = "🟡 MODERATE RISK"
            recommendation = "Enhanced monitoring and resource allocation recommended"
        elif prediction < 50000:
            risk = "🔴 HIGH RISK"
            recommendation = "Intensive intervention and resource mobilization needed"
        else:
            risk = "🔴🔴 VERY HIGH RISK"
            recommendation = "Emergency intervention and external support required"

        col_risk, col_rec = st.columns(2)
        with col_risk:
            st.write(f"**Risk Level:** {risk}")
        with col_rec:
            st.write(f"**Recommendation:** {recommendation}")

        # ─────────────────────────────────────────
        # PREDICTION REPORT TABLE
        # ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Prediction Report")

        report_df = pd.DataFrame({
            "Feature": [
                "State", "LGA", "Year", "Month",
                "Rainfall (mm)", "Temperature (°C)", "Vegetation Index",
                "Reporting Completeness (%)", "Timeliness (%)",
                "Season", "Intervention",
                "Predicted Malaria Cases", "Risk Level"
            ],
            "Value": [
                selected_state, selected_lga, year, selected_month_name,
                rainfall, temperature, vegetation,
                reporting, timeliness,
                season, intervention,
                f"{prediction:,.0f}", risk
            ]
        })

        st.dataframe(report_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        import traceback
        st.code(traceback.format_exc())

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>🦟 Malaria Prediction App | Built by Sunday Iyanu Samuel< small>
</div>
""", unsafe_allow_html=True)
