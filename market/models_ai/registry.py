from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

from .calibrator import (
    normalize_three_class_direction,
    normalize_three_class_quality,
)
from .trainer import get_model_artifact_paths
from ..engine.features_v2 import get_feature_columns_v2, build_features_v2


def load_registered_model(model_name):
    """
    Load a saved model and its metadata bundle.
    """
    model_path, scaler_path = get_model_artifact_paths(model_name)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    model = keras_load_model(model_path)
    metadata = joblib.load(scaler_path)

    return {
        "model": model,
        "metadata": metadata,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }


def prepare_latest_sequence(df, seq_length=24):
    """
    Prepare the latest feature window for inference.
    """
    feature_columns = get_feature_columns_v2()

    working_df = build_features_v2(df).copy()

    if len(working_df) < seq_length:
        raise ValueError("Not enough rows to build inference sequence.")

    latest_block = working_df[feature_columns].tail(seq_length).values.astype(np.float32)
    return working_df, latest_block


def infer_direction_v2(df, model_name="direction_model_v2"):
    """
    Run direction inference using the registered V2 direction model.
    """
    bundle = load_registered_model(model_name)
    model = bundle["model"]
    metadata = bundle["metadata"]

    seq_length = metadata["seq_length"]
    scaler = metadata["scaler"]
    feature_columns = metadata["feature_columns"]

    working_df = build_features_v2(df).copy()

    if len(working_df) < seq_length:
        raise ValueError("Not enough rows for direction inference.")

    latest_block = working_df[feature_columns].tail(seq_length).values.astype(np.float32)
    scaled_block = scaler.transform(latest_block)

    X = np.expand_dims(scaled_block, axis=0)
    probs = model.predict(X, verbose=0)[0]

    result = normalize_three_class_direction(probs)
    result["timestamp"] = str(working_df.iloc[-1]["time"])
    result["current_price"] = float(working_df.iloc[-1]["close"])

    return result


def infer_setup_quality_v2(df, model_name="setup_quality_model_v2"):
    """
    Run setup-quality inference using the registered V2 quality model.
    """
    bundle = load_registered_model(model_name)
    model = bundle["model"]
    metadata = bundle["metadata"]

    seq_length = metadata["seq_length"]
    scaler = metadata["scaler"]
    feature_columns = metadata["feature_columns"]

    working_df = build_features_v2(df).copy()

    if len(working_df) < seq_length:
        raise ValueError("Not enough rows for setup quality inference.")

    latest_block = working_df[feature_columns].tail(seq_length).values.astype(np.float32)
    scaled_block = scaler.transform(latest_block)

    X = np.expand_dims(scaled_block, axis=0)
    probs = model.predict(X, verbose=0)[0]

    result = normalize_three_class_quality(probs)
    result["timestamp"] = str(working_df.iloc[-1]["time"])
    result["current_price"] = float(working_df.iloc[-1]["close"])

    return result


if __name__ == "__main__":
    print("registry.py loaded successfully.")