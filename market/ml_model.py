import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb # type: ignore
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from .features import add_indicators

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FORECAST_HORIZON = 6
VALIDATION_RATIO = 0.2
MIN_TRAIN_ROWS = 80

FEATURE_COLUMNS = [
    "close",
    "rsi",
    "macd",
    "macd_signal",
    "sma_20",
    "ema_20",
    "volume_mean",
    "bb_high",
    "bb_low",
    "atr",
    "macd_hist",
    "price_vs_sma20",
    "price_vs_ema20",
    "bb_width",
    "volume_ratio",
    "candle_body_pct",
    "range_pct",
    "momentum_3h",
    "momentum_6h",
    "volatility_6h",
]


def get_feature_columns():
    """
    Return the feature list used by the ML models.
    """
    return FEATURE_COLUMNS.copy()


def get_model_paths(coin):
    """
    Return file paths for the classifier, regressor, and metadata.
    """
    classifier_path = os.path.join(MODELS_DIR, f"xgb_direction_{coin}.pkl")
    regressor_path = os.path.join(MODELS_DIR, f"xgb_return_{coin}.pkl")
    metadata_path = os.path.join(MODELS_DIR, f"xgb_meta_{coin}.json")
    return classifier_path, regressor_path, metadata_path


def add_ml_features(df):
    """
    Add higher-value engineered features for tree-based models.
    """
    df = df.copy()

    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["price_vs_ema20"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["close"]
    df["volume_ratio"] = df["volumeto"] / df["volume_mean"]
    df["candle_body_pct"] = (df["close"] - df["open"]) / df["open"]
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["momentum_3h"] = df["close"].pct_change(3)
    df["momentum_6h"] = df["close"].pct_change(6)
    df["volatility_6h"] = df["close"].pct_change().rolling(6).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def prepare_ml_dataset(df, coin, forecast_horizon=FORECAST_HORIZON):
    """
    Build a supervised learning dataset for a single coin.
    """
    coin_df = df[df["symbol"] == coin].copy()
    coin_df = add_indicators(coin_df)
    coin_df = add_ml_features(coin_df)

    coin_df["future_close"] = coin_df["close"].shift(-forecast_horizon)
    coin_df["target_return_pct"] = (
        (coin_df["future_close"] - coin_df["close"]) / coin_df["close"]
    ) * 100.0
    coin_df["target_return_pct"] = coin_df["target_return_pct"].clip(-20, 20)
    coin_df["target_direction"] = (coin_df["target_return_pct"] > 0).astype(int)

    coin_df = coin_df.dropna().reset_index(drop=True)

    if len(coin_df) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Not enough rows to train {coin}. Need at least {MIN_TRAIN_ROWS}, got {len(coin_df)}."
        )

    X = coin_df[FEATURE_COLUMNS].copy()
    y_direction = coin_df["target_direction"].copy()
    y_return = coin_df["target_return_pct"].copy()

    return coin_df, X, y_direction, y_return


def split_train_validation(X, y_direction, y_return, validation_ratio=VALIDATION_RATIO):
    """
    Perform a time-based train/validation split.
    """
    split_index = int(len(X) * (1 - validation_ratio))

    if split_index < 40 or (len(X) - split_index) < 10:
        raise ValueError("Dataset split is too small for reliable training and validation.")

    X_train = X.iloc[:split_index].copy()
    X_val = X.iloc[split_index:].copy()

    y_dir_train = y_direction.iloc[:split_index].copy()
    y_dir_val = y_direction.iloc[split_index:].copy()

    y_ret_train = y_return.iloc[:split_index].copy()
    y_ret_val = y_return.iloc[split_index:].copy()

    return X_train, X_val, y_dir_train, y_dir_val, y_ret_train, y_ret_val


def create_direction_model(y_train):
    """
    Create the primary direction model.
    """
    if y_train.nunique() < 2 or len(y_train) < MIN_TRAIN_ROWS:
        model = DummyClassifier(strategy="most_frequent")
        return model

    positive_ratio = float(y_train.mean())
    negative_ratio = 1.0 - positive_ratio
    scale_pos_weight = negative_ratio / positive_ratio if positive_ratio > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )
    return model


def create_return_model(y_train):
    """
    Create the primary return model.
    """
    if len(y_train) < MIN_TRAIN_ROWS:
        return DummyRegressor(strategy="mean")

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        objective="reg:squarederror",
        random_state=42,
    )
    return model


def evaluate_models(
    classifier,
    regressor,
    X_val,
    y_dir_val,
    y_ret_val,
):
    """
    Evaluate the classifier and regressor on the validation split.
    """
    dir_pred = classifier.predict(X_val)
    ret_pred = regressor.predict(X_val)

    direction_accuracy = accuracy_score(y_dir_val, dir_pred)
    return_mae = mean_absolute_error(y_ret_val, ret_pred)
    return_rmse = np.sqrt(mean_squared_error(y_ret_val, ret_pred))

    reg_direction = (ret_pred > 0).astype(int)
    return_direction_accuracy = accuracy_score(y_dir_val, reg_direction)

    return {
        "direction_accuracy": round(float(direction_accuracy), 4),
        "return_mae": round(float(return_mae), 4),
        "return_rmse": round(float(return_rmse), 4),
        "return_direction_accuracy": round(float(return_direction_accuracy), 4),
    }


def save_models(coin, classifier, regressor, metadata):
    """
    Save trained models and metadata to disk.
    """
    classifier_path, regressor_path, metadata_path = get_model_paths(coin)

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_models_for_coin(coin):
    """
    Load saved models and metadata for a coin.
    """
    classifier_path, regressor_path, metadata_path = get_model_paths(coin)

    if not (
        os.path.exists(classifier_path)
        and os.path.exists(regressor_path)
        and os.path.exists(metadata_path)
    ):
        return None, None, None

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return classifier, regressor, metadata


def train_xgb_for_coin(df, coin, forecast_horizon=FORECAST_HORIZON):
    """
    Train the XGBoost direction and return models for one coin.
    """
    prepared_df, X, y_direction, y_return = prepare_ml_dataset(
        df=df,
        coin=coin,
        forecast_horizon=forecast_horizon,
    )

    (
        X_train,
        X_val,
        y_dir_train,
        y_dir_val,
        y_ret_train,
        y_ret_val,
    ) = split_train_validation(X, y_direction, y_return)

    classifier = create_direction_model(y_dir_train)
    regressor = create_return_model(y_ret_train)

    classifier.fit(X_train, y_dir_train)
    regressor.fit(X_train, y_ret_train)

    metrics = evaluate_models(
        classifier=classifier,
        regressor=regressor,
        X_val=X_val,
        y_dir_val=y_dir_val,
        y_ret_val=y_ret_val,
    )

    metadata = {
        "coin": coin,
        "forecast_horizon": forecast_horizon,
        "feature_columns": FEATURE_COLUMNS,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "metrics": metrics,
    }

    save_models(coin, classifier, regressor, metadata)

    return {
        "classifier": classifier,
        "regressor": regressor,
        "metadata": metadata,
        "prepared_rows": len(prepared_df),
    }


def build_latest_feature_row(df, coin):
    """
    Build the latest feature row for live prediction.
    """
    coin_df = df[df["symbol"] == coin].copy()
    coin_df = add_indicators(coin_df)
    coin_df = add_ml_features(coin_df)

    if len(coin_df) == 0:
        raise ValueError(f"No feature-ready rows available for {coin}.")

    latest_row = coin_df.iloc[-1]
    X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS].to_dict()])

    return coin_df, X_latest


def build_expected_price_path(current_price, predicted_return_pct, steps=FORECAST_HORIZON):
    """
    Convert a total predicted return into a simple step path.
    """
    final_price = current_price * (1 + predicted_return_pct / 100.0)

    prices = []
    for step in range(1, steps + 1):
        ratio = step / steps
        step_price = current_price + ((final_price - current_price) * ratio)
        prices.append(float(step_price))

    return prices


def predict_xgb_for_coin(df, coin):
    """
    Run live inference using the saved XGBoost models.
    """
    classifier, regressor, metadata = load_models_for_coin(coin)

    if classifier is None or regressor is None:
        raise ValueError(f"No trained XGBoost models found for {coin}.")

    coin_df, X_latest = build_latest_feature_row(df, coin)
    latest_close = float(coin_df["close"].iloc[-1])

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(X_latest)[0]
        prob_down = float(probabilities[0])
        prob_up = float(probabilities[1])
    else:
        raw_pred = int(classifier.predict(X_latest)[0])
        prob_up = 1.0 if raw_pred == 1 else 0.0
        prob_down = 1.0 - prob_up

    predicted_return_pct = float(regressor.predict(X_latest)[0])
    expected_price = latest_close * (1 + predicted_return_pct / 100.0)
    forecast_path = build_expected_price_path(
        current_price=latest_close,
        predicted_return_pct=predicted_return_pct,
        steps=metadata.get("forecast_horizon", FORECAST_HORIZON) if metadata else FORECAST_HORIZON,
    )

    return {
        "current_price": latest_close,
        "prob_up": round(prob_up, 4),
        "prob_down": round(prob_down, 4),
        "predicted_return_pct": round(predicted_return_pct, 4),
        "expected_price": round(float(expected_price), 4),
        "forecast_path": forecast_path,
        "metadata": metadata,
    }