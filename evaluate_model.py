"""
evaluate_model.py
-----------------
Simple evaluation script for the trained disease risk XGBoost model.

What it does:
  1. Loads the same dataset used during training
  2. Rebuilds the training-style feature table
  3. Generates the same synthetic negative samples
  4. Loads the saved XGBoost model
  5. Prints accuracy, precision, recall, and F1 score
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

from train_ml_model import engineer_features, generate_negatives, load_positive_cases


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml_output" / "disease_xgb_model.json"
META_PATH = BASE_DIR / "ml_output" / "model_meta.json"
FEATURES = ["latitude", "longitude", "age", "gender", "month", "season"]


def load_evaluation_data():
    """Load and rebuild the same feature dataset used during training."""
    positive_raw = load_positive_cases()
    positive_features = engineer_features(positive_raw)
    negative_features = generate_negatives(positive_features)

    full_df = pd.concat([positive_features, negative_features], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return full_df


def load_label_encoder():
    """Restore label order from saved metadata when available."""
    encoder = LabelEncoder()

    if META_PATH.exists():
        with META_PATH.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        classes = meta.get("classes", [])
        if classes:
            encoder.classes_ = np.array(classes, dtype=object)
            return encoder

    encoder.fit(["Dengue", "Malaria", "Negative", "Typhoid"])
    return encoder


def load_model():
    """Load the trained XGBoost model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_ml_model.py first."
        )

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def main():
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    model = load_model()
    label_encoder = load_label_encoder()

    full_df = load_evaluation_data()
    full_df["label"] = label_encoder.transform(full_df["disease_label"])

    x_data = full_df[FEATURES].values
    y_true = full_df["label"].values
    y_pred = model.predict(x_data)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Samples evaluated : {len(full_df)}")
    print(f"Accuracy          : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
