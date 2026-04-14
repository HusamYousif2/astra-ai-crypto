import numpy as np
import pandas as pd

from ..features import add_indicators


def safe_divide(numerator, denominator):
    """
    Safely divide two values and return NaN when division is invalid.
    """
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def add_return_features(df):
    """
    Add return-based features across multiple horizons.
    """
    data = df.copy()

    data["ret_1"] = data["close"].pct_change(1)
    data["ret_3"] = data["close"].pct_change(3)
    data["ret_6"] = data["close"].pct_change(6)
    data["ret_12"] = data["close"].pct_change(12)
    data["ret_24"] = data["close"].pct_change(24)

    data["log_ret_1"] = np.log(data["close"] / data["close"].shift(1))
    data["log_ret_6"] = np.log(data["close"] / data["close"].shift(6))
    data["log_ret_24"] = np.log(data["close"] / data["close"].shift(24))

    return data


def add_price_structure_features(df):
    """
    Add candle-shape and price-structure features.
    """
    data = df.copy()

    data["candle_body"] = data["close"] - data["open"]
    data["candle_range"] = data["high"] - data["low"]
    data["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]

    data["body_to_range"] = data.apply(
        lambda row: safe_divide(abs(row["candle_body"]), row["candle_range"]),
        axis=1,
    )
    data["upper_wick_to_range"] = data.apply(
        lambda row: safe_divide(row["upper_wick"], row["candle_range"]),
        axis=1,
    )
    data["lower_wick_to_range"] = data.apply(
        lambda row: safe_divide(row["lower_wick"], row["candle_range"]),
        axis=1,
    )

    data["close_to_high_pct"] = (data["high"] - data["close"]) / data["close"]
    data["close_to_low_pct"] = (data["close"] - data["low"]) / data["close"]

    data["rolling_high_24"] = data["high"].rolling(24).max()
    data["rolling_low_24"] = data["low"].rolling(24).min()
    data["rolling_high_48"] = data["high"].rolling(48).max()
    data["rolling_low_48"] = data["low"].rolling(48).min()

    data["dist_to_24h_high"] = safe_series_divide(
        data["rolling_high_24"] - data["close"],
        data["close"],
    )
    data["dist_to_24h_low"] = safe_series_divide(
        data["close"] - data["rolling_low_24"],
        data["close"],
    )
    data["dist_to_48h_high"] = safe_series_divide(
        data["rolling_high_48"] - data["close"],
        data["close"],
    )
    data["dist_to_48h_low"] = safe_series_divide(
        data["close"] - data["rolling_low_48"],
        data["close"],
    )

    return data


def add_trend_features(df):
    """
    Add trend-location and slope features.
    """
    data = df.copy()

    data["ema_gap_pct"] = safe_series_divide(
        data["close"] - data["ema_20"],
        data["close"],
    )
    data["sma_gap_pct"] = safe_series_divide(
        data["close"] - data["sma_20"],
        data["close"],
    )

    data["ema_sma_gap_pct"] = safe_series_divide(
        data["ema_20"] - data["sma_20"],
        data["close"],
    )

    data["ema20_slope_3"] = safe_series_divide(
        data["ema_20"] - data["ema_20"].shift(3),
        data["ema_20"].shift(3),
    )
    data["ema20_slope_6"] = safe_series_divide(
        data["ema_20"] - data["ema_20"].shift(6),
        data["ema_20"].shift(6),
    )
    data["sma20_slope_6"] = safe_series_divide(
        data["sma_20"] - data["sma_20"].shift(6),
        data["sma_20"].shift(6),
    )

    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["macd_hist_change"] = data["macd_hist"].diff()

    return data


def add_volatility_features(df):
    """
    Add realized and relative volatility features.
    """
    data = df.copy()

    data["realized_vol_6"] = data["ret_1"].rolling(6).std()
    data["realized_vol_12"] = data["ret_1"].rolling(12).std()
    data["realized_vol_24"] = data["ret_1"].rolling(24).std()

    data["atr_pct"] = safe_series_divide(data["atr"], data["close"])

    data["range_pct"] = safe_series_divide(
        data["high"] - data["low"],
        data["close"],
    )

    data["range_mean_12"] = data["range_pct"].rolling(12).mean()
    data["range_mean_24"] = data["range_pct"].rolling(24).mean()

    data["volatility_ratio_6_24"] = safe_series_divide(
        data["realized_vol_6"],
        data["realized_vol_24"],
    )

    return data


def add_volume_features(df):
    """
    Add volume and participation features.
    """
    data = df.copy()

    volume_col = "volumeto" if "volumeto" in data.columns else "volume"

    data["volume_raw"] = data[volume_col]
    data["volume_change_1"] = data["volume_raw"].pct_change(1)
    data["volume_change_6"] = data["volume_raw"].pct_change(6)

    data["volume_ma_6"] = data["volume_raw"].rolling(6).mean()
    data["volume_ma_24"] = data["volume_raw"].rolling(24).mean()

    data["volume_vs_ma6"] = safe_series_divide(
        data["volume_raw"],
        data["volume_ma_6"],
    )
    data["volume_vs_ma24"] = safe_series_divide(
        data["volume_raw"],
        data["volume_ma_24"],
    )

    data["dollar_range_pressure"] = data["range_pct"] * data["volume_vs_ma24"]

    return data


def add_regime_features(df):
    """
    Add higher-level market regime features.
    """
    data = df.copy()

    data["trend_alignment_flag"] = (
        (data["close"] > data["ema_20"]) & (data["ema_20"] > data["sma_20"])
    ).astype(int)

    data["bear_alignment_flag"] = (
        (data["close"] < data["ema_20"]) & (data["ema_20"] < data["sma_20"])
    ).astype(int)

    data["rsi_bull_zone"] = ((data["rsi"] >= 55) & (data["rsi"] <= 70)).astype(int)
    data["rsi_bear_zone"] = ((data["rsi"] >= 30) & (data["rsi"] <= 45)).astype(int)

    data["compression_flag"] = (
        (data["range_mean_12"] < data["range_mean_24"])
        & (data["realized_vol_6"] < data["realized_vol_24"])
    ).astype(int)

    data["expansion_flag"] = (
        (data["range_mean_12"] > data["range_mean_24"])
        & (data["realized_vol_6"] > data["realized_vol_24"])
    ).astype(int)

    return data


def safe_series_divide(numerator, denominator):
    """
    Safely divide pandas Series objects.
    """
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def build_features_v2(df):
    """
    Build the next-generation feature set for one coin.
    """
    data = df.copy()

    required_columns = {"time", "open", "high", "low", "close"}
    if not required_columns.issubset(set(data.columns)):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    base_indicator_columns = {"rsi", "macd", "macd_signal", "sma_20", "ema_20", "atr"}
    if not base_indicator_columns.issubset(set(data.columns)):
        data = add_indicators(data)

    data = data.sort_values("time").reset_index(drop=True)

    data = add_return_features(data)
    data = add_price_structure_features(data)
    data = add_trend_features(data)
    data = add_volatility_features(data)
    data = add_volume_features(data)
    data = add_regime_features(data)

    data = data.dropna().reset_index(drop=True)
    return data


def build_features_v2_for_all_coins(df):
    """
    Build V2 features coin by coin, then merge all results.
    """
    if "symbol" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'symbol' column.")

    frames = []

    for symbol in df["symbol"].dropna().unique():
        coin_df = df[df["symbol"] == symbol].copy()
        feature_df = build_features_v2(coin_df)
        frames.append(feature_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_feature_columns_v2():
    """
    Return the main V2 feature columns used for training.
    """
    return [
        "close",
        "rsi",
        "macd",
        "macd_signal",
        "sma_20",
        "ema_20",
        "atr",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "log_ret_1",
        "log_ret_6",
        "log_ret_24",
        "candle_body",
        "candle_range",
        "upper_wick",
        "lower_wick",
        "body_to_range",
        "upper_wick_to_range",
        "lower_wick_to_range",
        "close_to_high_pct",
        "close_to_low_pct",
        "dist_to_24h_high",
        "dist_to_24h_low",
        "dist_to_48h_high",
        "dist_to_48h_low",
        "ema_gap_pct",
        "sma_gap_pct",
        "ema_sma_gap_pct",
        "ema20_slope_3",
        "ema20_slope_6",
        "sma20_slope_6",
        "macd_hist",
        "macd_hist_change",
        "realized_vol_6",
        "realized_vol_12",
        "realized_vol_24",
        "atr_pct",
        "range_pct",
        "range_mean_12",
        "range_mean_24",
        "volatility_ratio_6_24",
        "volume_raw",
        "volume_change_1",
        "volume_change_6",
        "volume_ma_6",
        "volume_ma_24",
        "volume_vs_ma6",
        "volume_vs_ma24",
        "dollar_range_pressure",
        "trend_alignment_flag",
        "bear_alignment_flag",
        "rsi_bull_zone",
        "rsi_bear_zone",
        "compression_flag",
        "expansion_flag",
    ]


if __name__ == "__main__":
    print("features_v2.py loaded successfully.")