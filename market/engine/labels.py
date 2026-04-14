import pandas as pd
import numpy as np

from ..features import add_indicators


DIRECTION_MAP = {
    "DOWN": 0,
    "NEUTRAL": 1,
    "UP": 2,
}

MOVE_BUCKET_MAP = {
    "TINY": 0,
    "SMALL": 1,
    "MEDIUM": 2,
    "LARGE": 3,
}

FOLLOW_THROUGH_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

SETUP_QUALITY_MAP = {
    "WEAK": 0,
    "MIXED": 1,
    "CLEAN": 2,
}


def safe_pct_change(current_price, future_price):
    """
    Compute percentage change safely.
    """
    if current_price == 0 or pd.isna(current_price) or pd.isna(future_price):
        return np.nan
    return ((future_price - current_price) / current_price) * 100.0


def classify_direction(future_return_pct, neutral_threshold=0.20):
    """
    Convert future return into a trading direction label.
    """
    if pd.isna(future_return_pct):
        return np.nan

    if future_return_pct > neutral_threshold:
        return "UP"
    if future_return_pct < -neutral_threshold:
        return "DOWN"
    return "NEUTRAL"


def classify_move_bucket(abs_return_pct):
    """
    Classify future move size into a practical bucket.
    """
    if pd.isna(abs_return_pct):
        return np.nan

    if abs_return_pct < 0.20:
        return "TINY"
    if abs_return_pct < 0.75:
        return "SMALL"
    if abs_return_pct < 1.75:
        return "MEDIUM"
    return "LARGE"


def classify_follow_through(future_return_pct, intrahorizon_return_pct):
    """
    Measure whether the move continued cleanly or faded.
    """
    if pd.isna(future_return_pct) or pd.isna(intrahorizon_return_pct):
        return np.nan

    future_abs = abs(future_return_pct)
    intra_abs = abs(intrahorizon_return_pct)

    if future_abs < 0.20:
        return "LOW"

    if intra_abs == 0:
        return "LOW"

    ratio = future_abs / intra_abs

    if ratio >= 0.85:
        return "HIGH"
    if ratio >= 0.50:
        return "MEDIUM"
    return "LOW"


def classify_setup_quality(row):
    """
    Estimate the structural cleanliness of a setup using current features.
    """
    try:
        score = 0.0

        close = row["close"]
        ema20 = row["ema_20"]
        sma20 = row["sma_20"]
        rsi = row["rsi"]
        macd = row["macd"]
        macd_signal = row["macd_signal"]
        atr = row["atr"]

        if close > ema20 > sma20:
            score += 2.0
        elif close < ema20 < sma20:
            score += 2.0

        if abs(macd - macd_signal) > 10:
            score += 1.0

        if 48 <= rsi <= 68:
            score += 1.0
        elif 32 <= rsi <= 48:
            score += 0.5

        if not pd.isna(atr) and close > 0:
            atr_pct = (atr / close) * 100
            if 0.25 <= atr_pct <= 2.50:
                score += 1.0

        if score >= 3.5:
            return "CLEAN"
        if score >= 2.0:
            return "MIXED"
        return "WEAK"

    except Exception:
        return np.nan


def encode_label(series, mapping):
    """
    Encode string labels into integer classes.
    """
    return series.map(mapping)


def build_direction_labels(df, horizon=6, neutral_threshold=0.20):
    """
    Build future direction labels using future close at a fixed horizon.
    """
    labeled = df.copy()

    labeled["future_close"] = labeled["close"].shift(-horizon)
    labeled["future_return_pct"] = labeled.apply(
        lambda row: safe_pct_change(row["close"], row["future_close"]),
        axis=1,
    )

    labeled["direction_label"] = labeled["future_return_pct"].apply(
        lambda x: classify_direction(x, neutral_threshold=neutral_threshold)
    )

    labeled["direction_target"] = encode_label(
        labeled["direction_label"], DIRECTION_MAP
    )

    return labeled


def build_move_labels(df):
    """
    Build future move-size labels from absolute future return.
    """
    labeled = df.copy()

    labeled["abs_future_return_pct"] = labeled["future_return_pct"].abs()
    labeled["move_bucket_label"] = labeled["abs_future_return_pct"].apply(
        classify_move_bucket
    )
    labeled["move_bucket_target"] = encode_label(
        labeled["move_bucket_label"], MOVE_BUCKET_MAP
    )

    return labeled


def build_follow_through_labels(df, horizon=6):
    """
    Build a follow-through label based on move persistence.
    """
    labeled = df.copy()

    half_horizon = max(1, horizon // 2)
    labeled["mid_close"] = labeled["close"].shift(-half_horizon)

    labeled["intrahorizon_return_pct"] = labeled.apply(
        lambda row: safe_pct_change(row["close"], row["mid_close"]),
        axis=1,
    )

    labeled["follow_through_label"] = labeled.apply(
        lambda row: classify_follow_through(
            row["future_return_pct"],
            row["intrahorizon_return_pct"],
        ),
        axis=1,
    )

    labeled["follow_through_target"] = encode_label(
        labeled["follow_through_label"], FOLLOW_THROUGH_MAP
    )

    return labeled


def build_setup_quality_labels(df):
    """
    Build a label that describes whether the current setup looks weak, mixed, or clean.
    """
    labeled = df.copy()

    labeled["setup_quality_label"] = labeled.apply(classify_setup_quality, axis=1)
    labeled["setup_quality_target"] = encode_label(
        labeled["setup_quality_label"], SETUP_QUALITY_MAP
    )

    return labeled


def build_training_labels(df, horizon=6, neutral_threshold=0.20):
    """
    Create a full supervised-label dataset for the next-generation AI pipeline.
    Expected input: a single-coin or multi-coin DataFrame with market columns.
    """
    labeled = df.copy()

    required_columns = {"close", "high", "low", "open", "volumeto"}
    if not required_columns.issubset(set(labeled.columns)):
        missing = required_columns - set(labeled.columns)
        raise ValueError(f"Missing required columns for labeling: {missing}")

    feature_columns = {"rsi", "macd", "macd_signal", "sma_20", "ema_20", "atr"}
    if not feature_columns.issubset(set(labeled.columns)):
        labeled = add_indicators(labeled)

    labeled = labeled.sort_values("time").reset_index(drop=True)

    labeled = build_direction_labels(
        labeled,
        horizon=horizon,
        neutral_threshold=neutral_threshold,
    )
    labeled = build_move_labels(labeled)
    labeled = build_follow_through_labels(labeled, horizon=horizon)
    labeled = build_setup_quality_labels(labeled)

    labeled = labeled.dropna(
        subset=[
            "future_close",
            "future_return_pct",
            "direction_label",
            "move_bucket_label",
            "follow_through_label",
            "setup_quality_label",
        ]
    ).reset_index(drop=True)

    return labeled


def build_training_labels_for_all_coins(df, horizon=6, neutral_threshold=0.20):
    """
    Build labels coin by coin, then merge them into one training dataset.
    """
    if "symbol" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'symbol' column.")

    frames = []

    for symbol in df["symbol"].dropna().unique():
        coin_df = df[df["symbol"] == symbol].copy()
        coin_labels = build_training_labels(
            coin_df,
            horizon=horizon,
            neutral_threshold=neutral_threshold,
        )
        frames.append(coin_labels)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("labels.py loaded successfully.")