import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Input, GRU, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


def create_sequences_from_matrix(features, targets, seq_length=24):
    """
    Convert tabular time-series data into sequence windows.
    """
    X, y = [], []

    if len(features) != len(targets):
        raise ValueError("Features and targets must have the same length.")

    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length:i])
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_direction_gru_model(input_shape, num_classes=3):
    """
    Build a GRU classifier for market direction prediction.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.20),
        GRU(32),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_setup_quality_gru_model(input_shape, num_classes=3):
    """
    Build a GRU classifier for setup quality prediction.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(48, return_sequences=True),
        Dropout(0.15),
        GRU(24),
        Dropout(0.15),
        Dense(24, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def predict_class_probabilities(model, X):
    """
    Predict probability distribution for each class.
    """
    if X is None or len(X) == 0:
        return np.array([])

    probs = model.predict(X, verbose=0)
    return probs


def predict_top_class(model, X):
    """
    Predict the top class index.
    """
    probs = predict_class_probabilities(model, X)

    if probs.size == 0:
        return np.array([])

    return np.argmax(probs, axis=1)


if __name__ == "__main__":
    print("sequence_model.py loaded successfully.")