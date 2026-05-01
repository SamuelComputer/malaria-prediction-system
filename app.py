"""
Flask REST API for the malaria prediction system.

Endpoints
---------
GET  /health           – Health check
POST /predict          – Single-patient prediction
POST /predict/batch    – Batch predictions
"""

from flask import Flask, request, jsonify

from src.model import load_model
from src.predict import predict_single, predict_batch
from src.preprocessing import validate_input

app = Flask(__name__)

_MODEL_UNAVAILABLE = "Model unavailable. Run train.py to train and save the model first."

# Load model and scaler once at startup
try:
    _model, _scaler = load_model()
    _model_ready = True
except FileNotFoundError:
    _model = None
    _scaler = None
    _model_ready = False


@app.route("/health", methods=["GET"])
def health():
    """Return service health status."""
    if not _model_ready:
        return jsonify({"status": "unavailable", "error": _MODEL_UNAVAILABLE}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Predict malaria for a single patient.

    Expects a JSON body with patient features.
    Returns prediction (0/1), probability, and label.
    """
    if not _model_ready:
        return jsonify({"error": _MODEL_UNAVAILABLE}), 503

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        result = predict_single(data, _model, _scaler)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    return jsonify(result), 200


@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """Predict malaria for a list of patients.

    Expects a JSON array of patient feature objects.
    Returns a list of prediction results.
    """
    if not _model_ready:
        return jsonify({"error": _MODEL_UNAVAILABLE}), 503

    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, list):
        return jsonify({"error": "Request body must be a JSON array of patient objects"}), 400

    results = []
    for i, patient in enumerate(data):
        try:
            results.append(predict_single(patient, _model, _scaler))
        except ValueError as exc:
            results.append({"error": str(exc), "index": i})

    return jsonify(results), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
