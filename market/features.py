# market/features.py

import pandas as pd
import ta


def add_indicators(df):
    """
    Add technical indicators used by professional traders
    """

    df = df.copy()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()

    # Volume indicator
    df["volume_mean"] = df["volumeto"].rolling(window=20).mean()

    df = df.dropna()
    # Bollinger Bands
    df["bb_high"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bb_low"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()

    # ATR (volatility)
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"]
    ).average_true_range()
    return df

