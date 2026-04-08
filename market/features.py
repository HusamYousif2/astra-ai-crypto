import pandas as pd
import ta


def add_indicators(df):
    """
    Add technical indicators used by professional traders.
    """

    df = df.copy()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Moving averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["ema_20"] = df["close"].ewm(span=20).mean()

    # Volume average
    df["volume_mean"] = df["volumeto"].rolling(window=20).mean()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()

    # ATR
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"]
    )
    df["atr"] = atr.average_true_range()

    # Remove incomplete rows after all indicators are calculated
    df = df.dropna().reset_index(drop=True)

    return df