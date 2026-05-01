# Malaria Prediction System (Nigeria)

A machine learning-powered web application that predicts malaria cases across Local Government Areas (LGAs) in Nigeria using environmental, geographic, and health system indicators.

Built with Streamlit and trained using ensemble machine learning models including Random Forest and XGBoost.

---

## Live Demo
https://malaria-prediction-system.streamlit.app/

---

## Project Overview

Malaria remains a major public health challenge in Nigeria, influenced by environmental conditions, healthcare reporting quality, and seasonal variations.

This project applies machine learning to:
- Predict malaria case burden at LGA level
- Support early intervention planning
- Provide data-driven insights for public health decision-making

---

## Problem Statement

Given:
- Weather conditions (rainfall, temperature, vegetation index)
- Health reporting indicators
- Seasonal patterns
- Geographic location (State, LGA)
- Intervention status

Predict expected malaria cases for a given period.

---

## Machine Learning Pipeline

### Models Evaluated:
- Random Forest Regressor
- XGBoost Regressor (Best Performing Model)

### Best Model:
- **XGBoost Regressor**

### Performance:
- R² Score: **~0.80**
- RMSE: **~0.77**

---

## Features Used

### Environmental Features:
- Rainfall (mm)
- Temperature (°C)
- Vegetation Index

### Health System Features:
- Reporting Completeness (%)
- Timeliness (%)
- Intervention Flag

### Temporal Features:
- Year
- Month
- Season (Dry/Rainy)

### Geographic Features:
- State
- Local Government Area (LGA)

---

## Web Application Features

- Interactive Streamlit dashboard
- Real-time malaria prediction
- Risk classification:
  - 🟢 Low Risk
  - 🟡 Moderate Risk
  - 🔴 High Risk
  - 🚨 Very High Risk
- Structured prediction report table

---

## Tech Stack

- Python 🐍
- Streamlit 🎈
- Pandas & NumPy 📊
- Scikit-learn 🤖
- XGBoost ⚡
- Matplotlib & Seaborn 📈
- Joblib 💾

---



## Project Structure
```
Malaria_Prediction_Project/
│
├── data_set/
│ ├── cleaned_malaria_burden_moderate_seed.csv
│ └── malaria_burden_moderate_seed.csv
│
├── deployment/
│ └── malaria_app.py
│
├── notebooks/
│ ├── 1_data_cleaning.ipynb
│ ├── 2_eda.ipynb
│ └── 3_model_training.ipynb
│
├── saved_model/
│ ├── malaria_model.pkl
│ ├── scaler.pkl
│ └── feature_names.pkl
│
├── requirements.txt
└── README.md
```


##  How to Run Locally

### 1. Clone repository
```bash
git clone https://github.com/your-username/malaria-prediction.git
cd malaria-prediction


pip install -r requirements.txt


streamlit run deployment/malaria_app.py

