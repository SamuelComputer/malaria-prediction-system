# Malaria Prediction System

A machine-learning-based system for predicting malaria infection based on patient symptoms, demographics, and environmental factors.

## Features

- **Random Forest classifier** trained on patient symptom and environmental data
- **REST API** (Flask) for real-time predictions
- **Data preprocessing pipeline** with feature engineering
- **Model evaluation** with accuracy, precision, recall, F1-score, and ROC-AUC

## Project Structure

```
malaria-prediction-system/
├── README.md
├── requirements.txt
├── train.py                    # Entry-point: train and save the model
├── app.py                      # Flask REST API
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Data cleaning and feature engineering
│   ├── model.py                # Model training, evaluation, persistence
│   └── predict.py              # Prediction utilities
├── data/
│   └── generate_sample_data.py # Generates a synthetic training dataset
├── models/                     # Saved model artifacts (created at runtime)
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    └── test_model.py
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python data/generate_sample_data.py
```

### 3. Train the model

```bash
python train.py
```

### 4. Start the API server

```bash
python app.py
```

### 5. Make a prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "gender": "M",
    "body_temperature": 38.9,
    "has_fever": 1,
    "has_chills": 1,
    "has_headache": 1,
    "has_nausea": 0,
    "has_fatigue": 1,
    "has_sweating": 1,
    "has_muscle_pain": 0,
    "region": "tropical",
    "season": "wet",
    "rainfall_mm": 120
  }'
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/predict` | Predict malaria (single patient) |
| `POST` | `/predict/batch` | Predict malaria (multiple patients) |

### Request schema (`/predict`)

| Field | Type | Description |
|-------|------|-------------|
| `age` | int | Patient age (years) |
| `gender` | str | `"M"` or `"F"` |
| `body_temperature` | float | Temperature in °C |
| `has_fever` | int | 0 or 1 |
| `has_chills` | int | 0 or 1 |
| `has_headache` | int | 0 or 1 |
| `has_nausea` | int | 0 or 1 |
| `has_fatigue` | int | 0 or 1 |
| `has_sweating` | int | 0 or 1 |
| `has_muscle_pain` | int | 0 or 1 |
| `region` | str | `"tropical"`, `"subtropical"`, or `"temperate"` |
| `season` | str | `"wet"` or `"dry"` |
| `rainfall_mm` | float | Monthly rainfall in mm |

### Response schema

```json
{
  "prediction": 1,
  "probability": 0.87,
  "label": "Malaria Positive"
}
```

## Running Tests

```bash
pytest tests/
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 14 input features (demographics, symptoms, environment, derived symptom count)
- **Target**: Binary classification (malaria positive / negative)
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC
