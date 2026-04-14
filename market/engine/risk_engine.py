import numpy as np
import pandas as pd


RISK_LEVELS = {
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
    "EXTREME": "Extreme",
}

TRADE_SAFETY_LEVELS = {
    "SAFE": "Safe",
    "CAUTION": "Caution",
    "UNSAFE": "Unsafe",
}


def safe_pct(value, base):
    """
    Compute percentage safely.
    """
    if base is None or base == 0 or pd.isna(base) or pd.isna(value):
        return np.nan
    return (value / base) * 100.0


def compute_drawdown_pressure(row):
    """
    Estimate recent downside pressure from local structure.
    """
    close = row.get("close", np.nan)
    rolling_low_24 = row.get("rolling_low_24", np.nan)
    rolling_high_24 = row.get("rolling_high_24", np.nan)

    if any(pd.isna(x) for x in [close, rolling_low_24, rolling_high_24]):
        return np.nan

    recent_range = rolling_high_24 - rolling_low_24
    if recent_range <= 0:
        return np.nan

    distance_to_low = close - rolling_low_24
    position_in_range = distance_to_low / recent_range

    # Lower values mean price is closer to the 24h low
    return round(float(1 - position_in_range), 4)


def compute_volatility_risk(row):
    """
    Estimate volatility-related risk.
    """
    atr_pct = row.get("atr_pct", np.nan)
    realized_vol_24 = row.get("realized_vol_24", np.nan)
    volatility_ratio_6_24 = row.get("volatility_ratio_6_24", np.nan)

    risk_score = 0.0

    if not pd.isna(atr_pct):
        if atr_pct >= 0.030:
            risk_score += 2.0
        elif atr_pct >= 0.018:
            risk_score += 1.0
        elif atr_pct >= 0.008:
            risk_score += 0.5

    if not pd.isna(realized_vol_24):
        if realized_vol_24 >= 0.025:
            risk_score += 1.5
        elif realized_vol_24 >= 0.015:
            risk_score += 0.75

    if not pd.isna(volatility_ratio_6_24):
        if volatility_ratio_6_24 >= 1.50:
            risk_score += 1.0
        elif volatility_ratio_6_24 >= 1.20:
            risk_score += 0.5

    return round(float(risk_score), 3)


def compute_trend_failure_risk(row):
    """
    Measure how fragile the current trend structure looks.
    """
    close = row.get("close", np.nan)
    ema20 = row.get("ema_20", np.nan)
    sma20 = row.get("sma_20", np.nan)
    macd = row.get("macd", np.nan)
    macd_signal = row.get("macd_signal", np.nan)
    rsi = row.get("rsi", np.nan)

    risk_score = 0.0

    if any(pd.isna(x) for x in [close, ema20, sma20, macd, macd_signal, rsi]):
        return np.nan

    if close < ema20:
        risk_score += 1.0

    if ema20 < sma20:
        risk_score += 1.0

    if macd < macd_signal:
        risk_score += 1.0

    if rsi < 45:
        risk_score += 0.75
    elif rsi > 75:
        risk_score += 0.5

    return round(float(risk_score), 3)


def compute_liquidity_stress(row):
    """
    Estimate whether the market is trading with weak participation.
    """
    volume_vs_ma24 = row.get("volume_vs_ma24", np.nan)
    range_pct = row.get("range_pct", np.nan)
    body_to_range = row.get("body_to_range", np.nan)

    risk_score = 0.0

    if not pd.isna(volume_vs_ma24):
        if volume_vs_ma24 < 0.65:
            risk_score += 1.0
        elif volume_vs_ma24 < 0.85:
            risk_score += 0.5

    if not pd.isna(range_pct) and not pd.isna(body_to_range):
        if range_pct > 0.02 and body_to_range < 0.35:
            risk_score += 0.75

    return round(float(risk_score), 3)


def compute_invalidation_risk(row):
    """
    Aggregate invalidation pressure from multiple sources.
    """
    drawdown_pressure = compute_drawdown_pressure(row)
    volatility_risk = compute_volatility_risk(row)
    trend_failure_risk = compute_trend_failure_risk(row)
    liquidity_stress = compute_liquidity_stress(row)

    total = 0.0

    if not pd.isna(drawdown_pressure):
        total += drawdown_pressure * 2.0

    if not pd.isna(volatility_risk):
        total += volatility_risk

    if not pd.isna(trend_failure_risk):
        total += trend_failure_risk

    if not pd.isna(liquidity_stress):
        total += liquidity_stress

    return round(float(total), 3)


def classify_risk_level(row):
    """
    Classify overall market risk level.
    """
    invalidation_risk = row.get("invalidation_risk", np.nan)
    volatility_risk = row.get("volatility_risk", np.nan)

    if pd.isna(invalidation_risk) and pd.isna(volatility_risk):
        return RISK_LEVELS["MEDIUM"]

    combined = 0.0
    if not pd.isna(invalidation_risk):
        combined += invalidation_risk
    if not pd.isna(volatility_risk):
        combined += volatility_risk * 0.75

    if combined >= 6.0:
        return RISK_LEVELS["EXTREME"]
    if combined >= 4.0:
        return RISK_LEVELS["HIGH"]
    if combined >= 2.0:
        return RISK_LEVELS["MEDIUM"]
    return RISK_LEVELS["LOW"]


def classify_trade_safety(row):
    """
    Convert risk conditions into a practical trade safety label.
    """
    risk_level = row.get("risk_level", RISK_LEVELS["MEDIUM"])
    market_state_label = row.get("market_state_label", "Transitional")
    setup_quality_label = row.get("setup_quality_label", "MIXED")

    if risk_level in {RISK_LEVELS["HIGH"], RISK_LEVELS["EXTREME"]}:
        return TRADE_SAFETY_LEVELS["UNSAFE"]

    if market_state_label in {"Transitional", "Volatile Expansion"}:
        return TRADE_SAFETY_LEVELS["CAUTION"]

    if setup_quality_label == "WEAK":
        return TRADE_SAFETY_LEVELS["CAUTION"]

    return TRADE_SAFETY_LEVELS["SAFE"]


def compute_risk_score(row):
    """
    Produce a normalized 0-100 risk score.
    Higher means more dangerous conditions.
    """
    invalidation_risk = row.get("invalidation_risk", np.nan)
    volatility_risk = row.get("volatility_risk", np.nan)
    trend_failure_risk = row.get("trend_failure_risk", np.nan)
    liquidity_stress = row.get("liquidity_stress", np.nan)

    raw = 0.0

    for value, weight in [
        (invalidation_risk, 1.0),
        (volatility_risk, 0.9),
        (trend_failure_risk, 0.8),
        (liquidity_stress, 0.6),
    ]:
        if not pd.isna(value):
            raw += value * weight

    score = max(min((raw / 8.0) * 100.0, 100.0), 0.0)
    return round(float(score), 1)


def add_risk_columns(df):
    """
    Add structured risk columns to a feature/state DataFrame.
    """
    data = df.copy()

    data["drawdown_pressure"] = data.apply(compute_drawdown_pressure, axis=1)
    data["volatility_risk"] = data.apply(compute_volatility_risk, axis=1)
    data["trend_failure_risk"] = data.apply(compute_trend_failure_risk, axis=1)
    data["liquidity_stress"] = data.apply(compute_liquidity_stress, axis=1)
    data["invalidation_risk"] = data.apply(compute_invalidation_risk, axis=1)
    data["risk_score"] = data.apply(compute_risk_score, axis=1)
    data["risk_level"] = data.apply(classify_risk_level, axis=1)
    data["trade_safety"] = data.apply(classify_trade_safety, axis=1)

    return data


def summarize_risk_snapshot(row):
    """
    Build a readable summary of the current risk state.
    """
    return {
        "risk_score": row.get("risk_score", np.nan),
        "risk_level": row.get("risk_level", RISK_LEVELS["MEDIUM"]),
        "trade_safety": row.get("trade_safety", TRADE_SAFETY_LEVELS["CAUTION"]),
        "invalidation_risk": row.get("invalidation_risk", np.nan),
        "volatility_risk": row.get("volatility_risk", np.nan),
        "trend_failure_risk": row.get("trend_failure_risk", np.nan),
        "liquidity_stress": row.get("liquidity_stress", np.nan),
    }


def get_latest_risk_snapshot(df):
    """
    Return the latest risk snapshot from a DataFrame that already contains risk columns.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    return summarize_risk_snapshot(latest)


if __name__ == "__main__":
    print("risk_engine.py loaded successfully.")