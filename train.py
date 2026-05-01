"""
Entry-point script: generate data (if needed), train the model, and save it.

Usage
-----
    python train.py
"""

import os
import sys
import pandas as pd

from data.generate_sample_data import generate, OUTPUT_PATH
from src.model import train, save_model

DATA_PATH = OUTPUT_PATH


def main() -> None:
    # Generate data if not already present
    if not os.path.exists(DATA_PATH):
        print("No dataset found. Generating synthetic data...")
        df = generate()
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset saved to {DATA_PATH}\n")
    else:
        print(f"Loading dataset from {DATA_PATH}\n")
        df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape : {df.shape}")
    print(f"Positive cases: {df['malaria_positive'].sum()} ({100 * df['malaria_positive'].mean():.1f}%)\n")

    print("Training Random Forest model...")
    model, scaler, metrics = train(df)

    print("\n=== Evaluation on held-out test set ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])

    save_model(model, scaler)
    print("Model and scaler saved to models/")


if __name__ == "__main__":
    main()
