import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore

from .sequence_model import (
    create_sequences_from_matrix,
    build_direction_gru_model,
    build_setup_quality_gru_model,
)
from ..engine.features_v2 import get_feature_columns_v2
from ..engine.pipeline_v2 import prepare_training_dataset_v2


MODELS_AI_DIR = Path(__file__).resolve().parent / "artifacts"
MODELS_AI_DIR.mkdir(parents=True, exist_ok=True)


def get_model_artifact_paths(model_name):
    """
    Return paths for model and scaler artifacts.
    """
    model_path = MODELS_AI_DIR / f"{model_name}.keras"
    scaler_path = MODELS_AI_DIR / f"{model_name}_scaler.pkl"

    return model_path, scaler_path


def clean_training_dataframe(df, feature_columns, target_column):
    """
    Keep only valid rows for training.
    """
    working_df = df.copy()

    working_df = working_df.dropna(subset=feature_columns + [target_column]).copy()
    working_df[target_column] = working_df[target_column].astype(int)

    return working_df.reset_index(drop=True)


def compute_balanced_class_weights(y):
    """
    Compute balanced class weights for imbalanced targets.
    """
    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def build_common_callbacks(model_path):
    """
    Build a standard callback set for training stability.
    """
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def prepare_training_matrices(df, target_column, seq_length=24):
    """
    Prepare X/y matrices for sequence training.
    """
    feature_columns = get_feature_columns_v2()

    required_columns = set(feature_columns + [target_column])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    working_df = clean_training_dataframe(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    X_raw = working_df[feature_columns].values.astype(np.float32)
    y_raw = working_df[target_column].values.astype(np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = create_sequences_from_matrix(
        X_scaled,
        y_raw,
        seq_length=seq_length,
    )

    if len(X_seq) == 0:
        raise ValueError(f"No training sequences were created for target: {target_column}")

    return X_seq, y_seq, scaler, feature_columns


def split_sequence_data(X_seq, y_seq):
    """
    Split sequence data into train and validation sets.
    """
    return train_test_split(
        X_seq,
        y_seq,
        test_size=0.2,
        random_state=42,
        stratify=y_seq,
    )


def train_direction_model(df, seq_length=24, epochs=12, batch_size=32):
    """
    Train a GRU model for direction classification.
    """
    X_seq, y_seq, scaler, feature_columns = prepare_training_matrices(
        df=df,
        target_column="direction_target",
        seq_length=seq_length,
    )

    X_train, X_val, y_train, y_val = split_sequence_data(X_seq, y_seq)
    class_weights = compute_balanced_class_weights(y_train)

    model = build_direction_gru_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=3,
    )

    model_path, scaler_path = get_model_artifact_paths("direction_model_v2")
    callbacks = build_common_callbacks(model_path)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "seq_length": seq_length,
            "target_column": "direction_target",
            "class_weights": class_weights,
        },
        scaler_path,
    )

    return {
        "model": model,
        "history": history.history,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "class_weights": class_weights,
    }


def train_setup_quality_model(df, seq_length=24, epochs=10, batch_size=32):
    """
    Train a GRU model for setup quality classification.
    """
    X_seq, y_seq, scaler, feature_columns = prepare_training_matrices(
        df=df,
        target_column="setup_quality_target",
        seq_length=seq_length,
    )

    X_train, X_val, y_train, y_val = split_sequence_data(X_seq, y_seq)
    class_weights = compute_balanced_class_weights(y_train)

    model = build_setup_quality_gru_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=3,
    )

    model_path, scaler_path = get_model_artifact_paths("setup_quality_model_v2")
    callbacks = build_common_callbacks(model_path)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    joblib.dump(
        {
            "scaler": scaler,
            "feature_columns": feature_columns,
            "seq_length": seq_length,
            "target_column": "setup_quality_target",
            "class_weights": class_weights,
        },
        scaler_path,
    )

    return {
        "model": model,
        "history": history.history,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "class_weights": class_weights,
    }


def build_and_prepare_training_data(fetch_all_coins_callable, horizon=6, neutral_threshold=0.20):
    """
    Fetch market data and prepare the V2 labeled dataset.
    """
    df = fetch_all_coins_callable()
    training_df = prepare_training_dataset_v2(
        df,
        horizon=horizon,
        neutral_threshold=neutral_threshold,
    )
    return training_df


if __name__ == "__main__":
    print("trainer.py loaded successfully.")