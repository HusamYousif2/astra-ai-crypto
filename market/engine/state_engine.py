import numpy as np
import pandas as pd


STATE_LABELS = {
    "BULLISH_MOMENTUM": "Bullish Momentum",
    "BEARISH_MOMENTUM": "Bearish Momentum",
    "RANGE_BOUND": "Range-Bound",
    "VOLATILE_EXPANSION": "Volatile Expansion",
    "COMPRESSED_BREAKOUT_SETUP": "Compressed Breakout Setup",
    "TRANSITIONAL": "Transitional",
}


def classify_volatility_state(row):
    """
    Classify the current volatility state.
    """
    vol_ratio = row.get("volatility_ratio_6_24", np.nan)
    atr_pct = row.get("atr_pct", np.nan)

    if pd.isna(vol_ratio) or pd.isna(atr_pct):
        return "UNKNOWN"

    if vol_ratio >= 1.35 or atr_pct >= 0.025:
        return "HIGH"
    if vol_ratio <= 0.85 and atr_pct <= 0.012:
        return "LOW"
    return "NORMAL"


def classify_trend_state(row):
    """
    Classify the current trend state.
    """
    close = row.get("close", np.nan)
    ema20 = row.get("ema_20", np.nan)
    sma20 = row.get("sma_20", np.nan)
    macd = row.get("macd", np.nan)
    macd_signal = row.get("macd_signal", np.nan)

    if any(pd.isna(x) for x in [close, ema20, sma20, macd, macd_signal]):
        return "UNKNOWN"

    if close > ema20 > sma20 and macd > macd_signal:
        return "UPTREND"

    if close < ema20 < sma20 and macd < macd_signal:
        return "DOWNTREND"

    return "MIXED"


def classify_momentum_state(row):
    """
    Classify momentum quality using RSI and MACD histogram.
    """
    rsi = row.get("rsi", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    macd_hist_change = row.get("macd_hist_change", np.nan)

    if any(pd.isna(x) for x in [rsi, macd_hist, macd_hist_change]):
        return "UNKNOWN"

    if rsi >= 55 and macd_hist > 0 and macd_hist_change >= 0:
        return "POSITIVE"

    if rsi <= 45 and macd_hist < 0 and macd_hist_change <= 0:
        return "NEGATIVE"

    return "MIXED"


def classify_structure_state(row):
    """
    Evaluate whether the market is compressed, expanding, or mixed.
    """
    compression_flag = row.get("compression_flag", 0)
    expansion_flag = row.get("expansion_flag", 0)
    range_mean_12 = row.get("range_mean_12", np.nan)
    range_mean_24 = row.get("range_mean_24", np.nan)

    if compression_flag == 1:
        return "COMPRESSED"

    if expansion_flag == 1:
        return "EXPANDING"

    if not pd.isna(range_mean_12) and not pd.isna(range_mean_24):
        if range_mean_12 < range_mean_24:
            return "TIGHTENING"
        if range_mean_12 > range_mean_24:
            return "WIDENING"

    return "MIXED"


def compute_state_score(row):
    """
    Compute a structured market-state score.
    """
    score = 0.0

    trend_state = classify_trend_state(row)
    momentum_state = classify_momentum_state(row)
    volatility_state = classify_volatility_state(row)
    structure_state = classify_structure_state(row)

    if trend_state == "UPTREND":
        score += 2.0
    elif trend_state == "DOWNTREND":
        score -= 2.0

    if momentum_state == "POSITIVE":
        score += 1.5
    elif momentum_state == "NEGATIVE":
        score -= 1.5

    if structure_state == "COMPRESSED":
        score += 0.25
    elif structure_state == "EXPANDING":
        score += 0.5

    if volatility_state == "HIGH":
        score += 0.25 if score > 0 else -0.25

    return round(float(score), 3)


def classify_market_state(row):
    """
    Produce a higher-level market state label.
    """
    trend_state = classify_trend_state(row)
    momentum_state = classify_momentum_state(row)
    volatility_state = classify_volatility_state(row)
    structure_state = classify_structure_state(row)

    if trend_state == "UPTREND" and momentum_state == "POSITIVE":
        return STATE_LABELS["BULLISH_MOMENTUM"]

    if trend_state == "DOWNTREND" and momentum_state == "NEGATIVE":
        return STATE_LABELS["BEARISH_MOMENTUM"]

    if structure_state == "COMPRESSED" and trend_state == "MIXED":
        return STATE_LABELS["COMPRESSED_BREAKOUT_SETUP"]

    if volatility_state == "HIGH" and structure_state in {"EXPANDING", "WIDENING"}:
        return STATE_LABELS["VOLATILE_EXPANSION"]

    rsi = row.get("rsi", np.nan)
    macd_hist = row.get("macd_hist", np.nan)
    if not pd.isna(rsi) and not pd.isna(macd_hist):
        if 45 <= rsi <= 55 and abs(macd_hist) < 15:
            return STATE_LABELS["RANGE_BOUND"]

    return STATE_LABELS["TRANSITIONAL"]


def compute_state_confidence(row):
    """
    Estimate how clean the current market state looks.
    """
    trend_state = classify_trend_state(row)
    momentum_state = classify_momentum_state(row)
    structure_state = classify_structure_state(row)

    confidence = 0.35

    if trend_state in {"UPTREND", "DOWNTREND"}:
        confidence += 0.20

    if momentum_state in {"POSITIVE", "NEGATIVE"}:
        confidence += 0.20

    if structure_state in {"COMPRESSED", "EXPANDING"}:
        confidence += 0.10

    atr_pct = row.get("atr_pct", np.nan)
    if not pd.isna(atr_pct):
        if 0.003 <= atr_pct <= 0.025:
            confidence += 0.10

    return round(float(min(confidence, 0.95)), 3)


def add_market_state_columns(df):
    """
    Add market-state columns to a feature DataFrame.
    """
    data = df.copy()

    data["trend_state"] = data.apply(classify_trend_state, axis=1)
    data["momentum_state"] = data.apply(classify_momentum_state, axis=1)
    data["volatility_state"] = data.apply(classify_volatility_state, axis=1)
    data["structure_state"] = data.apply(classify_structure_state, axis=1)
    data["market_state_score"] = data.apply(compute_state_score, axis=1)
    data["market_state_label"] = data.apply(classify_market_state, axis=1)
    data["market_state_confidence"] = data.apply(compute_state_confidence, axis=1)

    return data


def summarize_current_state(row):
    """
    Build a readable summary for the latest market state.
    """
    label = row.get("market_state_label", STATE_LABELS["TRANSITIONAL"])
    trend_state = row.get("trend_state", "UNKNOWN")
    momentum_state = row.get("momentum_state", "UNKNOWN")
    volatility_state = row.get("volatility_state", "UNKNOWN")
    structure_state = row.get("structure_state", "UNKNOWN")

    return {
        "market_state_label": label,
        "trend_state": trend_state,
        "momentum_state": momentum_state,
        "volatility_state": volatility_state,
        "structure_state": structure_state,
        "market_state_score": row.get("market_state_score", np.nan),
        "market_state_confidence": row.get("market_state_confidence", np.nan),
    }


def get_latest_state_snapshot(df):
    """
    Return the latest state snapshot from a DataFrame that already contains state columns.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    return summarize_current_state(latest)


if __name__ == "__main__":
    print("state_engine.py loaded successfully.")