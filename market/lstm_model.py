import os

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore

from .features import add_indicators

SEQ_LENGTH = 168

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def get_model_paths(coin):
    model_path = os.path.join(MODELS_DIR, f"lstm_{coin}.h5")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{coin}.pkl")
    return model_path, scaler_path


def create_sequences(data, seq_length):
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def build_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm_for_coin(df, coin):
    features = [
        "close",
        "rsi",
        "macd",
        "macd_signal",
        "sma_20",
        "ema_20",
        "volume_mean",
        "atr",
    ]

    coin_df = df[df["symbol"] == coin].copy()
    coin_df = add_indicators(coin_df)
    coin_df = coin_df.dropna().reset_index(drop=True)

    if len(coin_df) < (SEQ_LENGTH + 50):
        raise ValueError(
            f"Not enough data to train {coin}. Need at least {SEQ_LENGTH + 50} rows, got {len(coin_df)}."
        )

    data_matrix = coin_df[features].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_matrix)

    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    if len(X) == 0:
        raise ValueError(f"Not enough data to create sequences for {coin}.")

    split_index = int(len(X) * 0.8)

    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

    print(f"Training Deep Learning Model for {coin} with SEQ_LENGTH={SEQ_LENGTH}...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=12,
        batch_size=32,
        verbose=1,
    )

    model_path, scaler_path = get_model_paths(coin)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler