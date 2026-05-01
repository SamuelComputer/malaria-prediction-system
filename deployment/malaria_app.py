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
# TITLE
# ─────────────────────────────────────────────
st.title("🦟 Malaria Cases Prediction")

st.markdown("""
Predict malaria cases in Nigerian LGAs using:
- Environmental factors
- Geographic factors
- Seasonal indicators
- Intervention data
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
    st.error(f"Error loading files: {e}")
    st.stop()

# ─────────────────────────────────────────────
# GET STATES & LGAs
# ─────────────────────────────────────────────
states = []
lgas = []

for feature in all_features:

    if feature.startswith("state_"):
        states.append(feature.replace("state_", ""))

    elif feature.startswith("lga_"):
        lgas.append(feature.replace("lga_", ""))

states = sorted(list(set(states)))
lgas = sorted(list(set(lgas)))

# ─────────────────────────────────────────────
# MONTHS
# ─────────────────────────────────────────────
month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
st.header("📊 Input Features")

col1, col2 = st.columns(2)

# ─────────────────────────────────────────────
# COLUMN 1
# ─────────────────────────────────────────────
with col1:

    selected_state = st.selectbox(
        "Select State",
        states
    )

    selected_lga = st.selectbox(
        "Select LGA",
        lgas
    )

    year = st.number_input(
        "Year",
        min_value=2018,
        max_value=2030,
        value=2023
    )

    selected_month_name = st.selectbox(
        "Month",
        list(month_mapping.keys())
    )

    month = month_mapping[selected_month_name]

# ─────────────────────────────────────────────
# COLUMN 2
# ─────────────────────────────────────────────
with col2:

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=200.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=10.0,
        value=27.0
    )

    vegetation = st.number_input(
        "Vegetation Index",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    reporting = st.slider(
        "Reporting Completeness (%)",
        0,
        100,
        80
    )

    timeliness = st.slider(
        "Timeliness (%)",
        0,
        100,
        80
    )

# ─────────────────────────────────────────────
# EXTRA INPUTS
# ─────────────────────────────────────────────
season = st.selectbox(
    "Season",
    ["Dry", "Rainy"]
)

intervention = st.selectbox(
    "Intervention Applied?",
    ["No", "Yes"]
)

# ─────────────────────────────────────────────
# PREDICTION BUTTON
# ─────────────────────────────────────────────
predict_button = st.button(
    "Predict Malaria Cases",
    type="primary"
)

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_button:

    try:

        # Create dataframe with correct columns
        input_data = pd.DataFrame(
            np.zeros((1, len(all_features))),
            columns=all_features
        )

        # Numerical Features
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

        # One-hot encode state
        state_col = f"state_{selected_state}"

        if state_col in input_data.columns:
            input_data[state_col] = 1

        # One-hot encode LGA
        lga_col = f"lga_{selected_lga}"

        if lga_col in input_data.columns:
            input_data[lga_col] = 1

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # ─────────────────────────────────────────
        # OUTPUT
        # ─────────────────────────────────────────
        st.success("Prediction Successful ✅")

        st.metric(
            "Predicted Malaria Cases",
            f"{prediction:,.0f}"

            
        )        

         
        # ─────────────────────────────────────────
        # RISK ASSESSMENT
        # ─────────────────────────────────────────
        if prediction < 2500:
            risk = "LOW RISK"
        
        elif prediction < 11000:
            risk = "MODERATE RISK"
        
        elif prediction < 50000:
            risk = "HIGH RISK"
        
        else:
            risk = "VERY HIGH RISK"

        st.subheader("Risk Assessment")
        st.write(risk)      
        
        # ─────────────────────────────────────────
        # FORECAST VISUALIZATION
        # ─────────────────────────────────────────
        
        
        st.subheader("📋 Prediction Report")
        
        report_df = pd.DataFrame({
            "Data": [
                "State",
                "LGA",
                "Year",
                "Month",
                "Rainfall (mm)",
                "Temperature (°C)",
                "Vegetation Index",
                "Reporting Completeness (%)",
                "Timeliness (%)",
                "Season",
                "Intervention",
                "Predicted Malaria Cases",
                "Risk Level"
            ],
            "Value": [
                selected_state,
                selected_lga,
                year,
                selected_month_name,
                rainfall,
                temperature,
                vegetation,
                reporting,
                timeliness,
                season,
                intervention,
                f"{prediction:,.0f}",
                risk
            ]
        })
        
        st.dataframe(report_df, use_container_width=True, hide_index=True)            
        
        
        
      
    except Exception as e:

        st.error(f"Prediction Error: {e}")

        import traceback
        st.code(traceback.format_exc())
        
        # ─────────────────────────────────────────────
        # SIDEBAR
        # ─────────────────────────────────────────────
        st.sidebar.title("ℹ️ About")
        
        st.sidebar.markdown("""
        ### Malaria Prediction System
        
        Built using:
        - Streamlit
        - Scikit-learn
        - XGBoost
        
        ### Model Performance
        - R² Score: 0.801
- RMSE: 6,064

### Purpose
Predict malaria burden in Nigerian LGAs.
""")