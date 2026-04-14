import numpy as np
import pandas as pd

from .labels import classify_setup_quality


SIGNAL_LABELS = {
    "BUY": "BUY",
    "SELL": "SELL",
    "HOLD": "HOLD",
}

SIGNAL_STRENGTH_LABELS = {
    "WEAK": "Weak",
    "MODERATE": "Moderate",
    "STRONG": "Strong",
}

TRADE_STANCE_LABELS = {
    "NO_TRADE": "No Trade",
    "WATCH": "Watch Closely",
    "CANDIDATE": "Trade Candidate",
}

NEWS_IMPACT_LABELS = {
    "LOW": "Low",
    "MEDIUM": "Medium",
    "HIGH": "High",
}


def safe_value(value, default=np.nan):
    """
    Return a safe numeric value when possible.
    """
    if value is None or pd.isna(value):
        return default
    return value


def classify_signal_strength(abs_score):
    """
    Convert a decision score into a user-friendly strength label.
    """
    if abs_score >= 4.5:
        return SIGNAL_STRENGTH_LABELS["STRONG"]
    if abs_score >= 2.5:
        return SIGNAL_STRENGTH_LABELS["MODERATE"]
    return SIGNAL_STRENGTH_LABELS["WEAK"]


def classify_news_impact(news_context):
    """
    Estimate how important the current news flow is.
    """
    if not news_context:
        return NEWS_IMPACT_LABELS["LOW"]

    sentiment_score = abs(news_context.get("sentiment_score", 0.0))
    article_count = news_context.get("article_count", 0)

    impact_score = 0.0

    if sentiment_score >= 0.60:
        impact_score += 2.0
    elif sentiment_score >= 0.25:
        impact_score += 1.0
    elif sentiment_score >= 0.10:
        impact_score += 0.5

    if article_count >= 6:
        impact_score += 1.5
    elif article_count >= 3:
        impact_score += 1.0
    elif article_count >= 1:
        impact_score += 0.5

    if impact_score >= 2.5:
        return NEWS_IMPACT_LABELS["HIGH"]
    if impact_score >= 1.0:
        return NEWS_IMPACT_LABELS["MEDIUM"]
    return NEWS_IMPACT_LABELS["LOW"]


def detect_news_contradiction(news_context, market_state_label):
    """
    Detect whether news direction conflicts with market state.
    """
    if not news_context:
        return False

    sentiment_label = news_context.get("sentiment_label", "NEUTRAL")

    if sentiment_label == "BULLISH" and market_state_label in {"Bearish Momentum"}:
        return True

    if sentiment_label == "BEARISH" and market_state_label in {"Bullish Momentum"}:
        return True

    return False


def build_state_contribution(row):
    """
    Build decision contribution from market state.
    """
    score = 0.0
    bullish_factors = []
    bearish_factors = []
    watchpoints = []

    market_state_label = row.get("market_state_label", "Transitional")
    market_state_score = safe_value(row.get("market_state_score"), 0.0)
    market_state_confidence = safe_value(row.get("market_state_confidence"), 0.35)

    score += market_state_score * market_state_confidence

    if market_state_label == "Bullish Momentum":
        bullish_factors.append("Market state shows bullish momentum with constructive price structure.")
    elif market_state_label == "Bearish Momentum":
        bearish_factors.append("Market state shows bearish momentum with weak structure.")
    elif market_state_label == "Range-Bound":
        watchpoints.append("The market appears range-bound, so directional follow-through may stay limited.")
    elif market_state_label == "Volatile Expansion":
        bearish_factors.append("The market is in a volatile expansion phase, which raises instability.")
    elif market_state_label == "Compressed Breakout Setup":
        watchpoints.append("The market is compressed and may be preparing for a breakout.")
    else:
        watchpoints.append("The market remains transitional and does not yet show a clean state.")

    return score, bullish_factors, bearish_factors, watchpoints


def build_structure_contribution(row):
    """
    Build contribution from trend, momentum, and local structure.
    """
    score = 0.0
    bullish_factors = []
    bearish_factors = []
    watchpoints = []

    close = safe_value(row.get("close"))
    ema20 = safe_value(row.get("ema_20"))
    sma20 = safe_value(row.get("sma_20"))
    macd = safe_value(row.get("macd"))
    macd_signal = safe_value(row.get("macd_signal"))
    rsi = safe_value(row.get("rsi"))
    atr_pct = safe_value(row.get("atr_pct"))
    volume_vs_ma24 = safe_value(row.get("volume_vs_ma24"))

    if not any(pd.isna(x) for x in [close, ema20, sma20]):
        if close > ema20 > sma20:
            score += 1.5
            bullish_factors.append("Price is trading above key short-term trend averages.")
        elif close < ema20 < sma20:
            score -= 1.5
            bearish_factors.append("Price is trading below key short-term trend averages.")

    if not any(pd.isna(x) for x in [macd, macd_signal]):
        if macd > macd_signal:
            score += 1.0
            bullish_factors.append("MACD momentum remains supportive.")
        else:
            score -= 1.0
            bearish_factors.append("MACD momentum remains weak.")

    if not pd.isna(rsi):
        if 52 <= rsi <= 68:
            score += 0.75
            bullish_factors.append("RSI shows constructive momentum without being overstretched.")
        elif 45 <= rsi < 52:
            score += 0.25
            watchpoints.append("Momentum is only slightly positive and still needs confirmation.")
        elif 32 <= rsi < 45:
            score -= 0.5
            bearish_factors.append("RSI remains below neutral momentum territory.")
        elif rsi > 72:
            bearish_factors.append("RSI is elevated and raises short-term reversal risk.")
            watchpoints.append("Watch for momentum exhaustion after a strong upward stretch.")
        elif rsi < 28:
            bullish_factors.append("RSI is deeply stretched and may support a relief rebound.")
            watchpoints.append("Watch for rebound confirmation from oversold conditions.")

    if not pd.isna(atr_pct):
        if atr_pct > 0.03:
            bearish_factors.append("ATR is elevated, which increases execution risk.")
        elif 0.004 <= atr_pct <= 0.02:
            bullish_factors.append("Volatility remains within a workable range for directional setups.")

    if not pd.isna(volume_vs_ma24):
        if volume_vs_ma24 > 1.15:
            bullish_factors.append("Participation is stronger than the recent average.")
            score += 0.5
        elif volume_vs_ma24 < 0.80:
            bearish_factors.append("Participation remains soft relative to recent average.")
            score -= 0.35

    return score, bullish_factors, bearish_factors, watchpoints


def build_news_contribution(news_context, market_state_label):
    """
    Build decision contribution from news and sentiment.
    """
    score = 0.0
    bullish_factors = []
    bearish_factors = []
    watchpoints = []

    if not news_context:
        watchpoints.append("No meaningful news context is currently available.")
        return score, bullish_factors, bearish_factors, watchpoints

    sentiment_score = news_context.get("sentiment_score", 0.0)
    sentiment_label = news_context.get("sentiment_label", "NEUTRAL")
    article_count = news_context.get("article_count", 0)
    news_impact = classify_news_impact(news_context)

    if sentiment_score > 0.25:
        score += 0.9
        bullish_factors.append("News flow is supportive overall.")
    elif sentiment_score > 0.10:
        score += 0.35
        bullish_factors.append("News flow is modestly supportive.")
    elif sentiment_score < -0.25:
        score -= 0.9
        bearish_factors.append("News flow is negative overall.")
    elif sentiment_score < -0.10:
        score -= 0.35
        bearish_factors.append("News flow is modestly negative.")
    else:
        watchpoints.append("News flow is mixed and not strongly directional.")

    if article_count >= 4:
        watchpoints.append("There is enough recent coverage for news to matter in the current setup.")

    if news_impact == "High":
        watchpoints.append("News impact is high and may accelerate market reaction.")

    if detect_news_contradiction(news_context, market_state_label):
        bearish_factors.append("News direction conflicts with the current market structure.")
        score -= 0.4

    return score, bullish_factors, bearish_factors, watchpoints


def build_risk_contribution(row):
    """
    Build decision penalties from risk conditions.
    """
    score = 0.0
    bullish_factors = []
    bearish_factors = []
    watchpoints = []

    risk_score = safe_value(row.get("risk_score"), 50.0)
    risk_level = row.get("risk_level", "Medium")
    trade_safety = row.get("trade_safety", "Caution")
    invalidation_risk = safe_value(row.get("invalidation_risk"), 0.0)

    if risk_level == "Extreme":
        score -= 2.0
        bearish_factors.append("Risk conditions are extreme and unfavorable for clean execution.")
    elif risk_level == "High":
        score -= 1.4
        bearish_factors.append("Risk conditions are elevated and reduce setup quality.")
    elif risk_level == "Medium":
        score -= 0.6
        watchpoints.append("Risk conditions are manageable but still require discipline.")
    else:
        bullish_factors.append("Risk conditions remain relatively contained.")

    if trade_safety == "Unsafe":
        score -= 1.0
        bearish_factors.append("Trade safety is low under the current market conditions.")
    elif trade_safety == "Caution":
        score -= 0.35
        watchpoints.append("Current conditions call for extra caution.")
    else:
        bullish_factors.append("Trade safety looks acceptable for monitored execution.")

    if invalidation_risk >= 4.5:
        bearish_factors.append("Invalidation pressure is high if the setup fails.")
    elif invalidation_risk >= 2.5:
        watchpoints.append("The setup has moderate invalidation pressure and needs close monitoring.")

    if risk_score >= 70:
        watchpoints.append("Position sizing should remain conservative until conditions improve.")

    return score, bullish_factors, bearish_factors, watchpoints


def classify_signal(final_score):
    """
    Convert the final decision score into a signal.
    """
    if final_score >= 2.5:
        return SIGNAL_LABELS["BUY"]
    if final_score <= -2.5:
        return SIGNAL_LABELS["SELL"]
    return SIGNAL_LABELS["HOLD"]


def determine_trade_stance(signal, signal_strength, trade_safety, news_impact):
    """
    Convert decision state into a practical trade stance.
    """
    if signal == "HOLD":
        return TRADE_STANCE_LABELS["NO_TRADE"]

    if signal_strength == SIGNAL_STRENGTH_LABELS["WEAK"]:
        return TRADE_STANCE_LABELS["WATCH"]

    if trade_safety == "Unsafe":
        return TRADE_STANCE_LABELS["WATCH"]

    if news_impact == NEWS_IMPACT_LABELS["HIGH"]:
        return TRADE_STANCE_LABELS["WATCH"]

    return TRADE_STANCE_LABELS["CANDIDATE"]


def compute_setup_score(final_score, row):
    """
    Convert raw decision conditions into a 0-100 setup score.
    """
    risk_score = safe_value(row.get("risk_score"), 50.0)
    base_score = ((final_score + 6.0) / 12.0) * 100.0
    adjusted = base_score - (risk_score * 0.18)

    return round(float(max(min(adjusted, 100.0), 0.0)), 1)


def build_signal_explainer(signal, signal_strength, market_state_label):
    """
    Create a readable explanation of what the current signal means.
    """
    if signal == "BUY":
        return (
            f"The AI sees a {signal_strength.lower()} bullish setup. "
            f"The current market state is {market_state_label.lower()}, so the bias is positive but still needs disciplined execution."
        )

    if signal == "SELL":
        return (
            f"The AI sees a {signal_strength.lower()} bearish setup. "
            f"The current market state is {market_state_label.lower()}, so defensive positioning is more appropriate right now."
        )

    return (
        f"The AI does not see a clean trade signal yet. "
        f"The current market state is {market_state_label.lower()}, which suggests waiting for better alignment."
    )


def build_confidence_explainer():
    """
    Explain what the score means for end users.
    """
    return (
        "This score reflects the quality and clarity of the current trading setup. "
        "It does not represent certainty about the current market price."
    )


def deduplicate_items(items, limit=4):
    """
    Keep unique readable list items in order.
    """
    clean = []
    seen = set()

    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            clean.append(item)

    return clean[:limit]


def build_decision_snapshot(row, news_context=None, forecast_context=None):
    """
    Build a full decision snapshot for one market row.
    """
    working_row = row.copy()

    if "setup_quality_label" not in working_row or pd.isna(working_row.get("setup_quality_label")):
        working_row["setup_quality_label"] = classify_setup_quality(working_row)

    market_state_label = working_row.get("market_state_label", "Transitional")
    trade_safety = working_row.get("trade_safety", "Caution")

    total_score = 0.0
    bullish_factors = []
    bearish_factors = []
    watchpoints = []

    for builder in [
        lambda: build_state_contribution(working_row),
        lambda: build_structure_contribution(working_row),
        lambda: build_news_contribution(news_context, market_state_label),
        lambda: build_risk_contribution(working_row),
    ]:
        contribution_score, bulls, bears, watches = builder()
        total_score += contribution_score
        bullish_factors.extend(bulls)
        bearish_factors.extend(bears)
        watchpoints.extend(watches)

    forecast_direction = None
    expected_move_pct = np.nan

    if forecast_context:
        expected_move_pct = forecast_context.get("expected_move_pct", np.nan)
        forecast_direction = forecast_context.get("direction", None)

        if not pd.isna(expected_move_pct):
            if expected_move_pct > 0.75:
                total_score += 1.0
                bullish_factors.append("Forecast context points to meaningful upside.")
            elif expected_move_pct > 0.20:
                total_score += 0.4
                bullish_factors.append("Forecast context remains modestly positive.")
            elif expected_move_pct < -0.75:
                total_score -= 1.0
                bearish_factors.append("Forecast context points to meaningful downside.")
            elif expected_move_pct < -0.20:
                total_score -= 0.4
                bearish_factors.append("Forecast context remains modestly negative.")

    signal = classify_signal(total_score)
    signal_strength = classify_signal_strength(abs(total_score))
    news_impact = classify_news_impact(news_context)
    trade_stance = determine_trade_stance(
        signal=signal,
        signal_strength=signal_strength,
        trade_safety=trade_safety,
        news_impact=news_impact,
    )
    setup_score = compute_setup_score(total_score, working_row)

    if signal == "BUY":
        direction = "UP"
    elif signal == "SELL":
        direction = "DOWN"
    else:
        if forecast_direction in {"UP", "DOWN"}:
            direction = forecast_direction
        else:
            direction = "UP" if total_score >= 0 else "DOWN"

    if signal == "BUY":
        ai_summary = (
            "The market setup is constructive. Trend, structure, and context support a bullish bias, "
            "but execution quality still depends on risk staying under control."
        )
    elif signal == "SELL":
        ai_summary = (
            "The market setup is fragile. Weak structure and risk conditions support a defensive bearish view."
        )
    else:
        ai_summary = (
            "The market does not show a clean edge right now. Conditions look mixed, so patience is more valuable than forcing a trade."
        )

    if working_row.get("setup_quality_label") == "WEAK":
        watchpoints.append("Setup quality is weak, so confirmation should come before any aggressive action.")
    elif working_row.get("setup_quality_label") == "MIXED":
        watchpoints.append("Setup quality is mixed, so watch for stronger alignment before acting.")
    else:
        bullish_factors.append("Setup quality appears relatively clean.")

    if trade_stance == TRADE_STANCE_LABELS["WATCH"]:
        watchpoints.append("This is better treated as a monitored setup than an immediate trade.")

    decision_snapshot = {
        "signal": signal,
        "direction": direction,
        "signal_strength": signal_strength,
        "trade_stance": trade_stance,
        "news_impact": news_impact,
        "setup_score": setup_score,
        "decision_score": round(float(total_score), 3),
        "market_regime": market_state_label,
        "risk_level": working_row.get("risk_level", "Medium"),
        "trade_safety": trade_safety,
        "setup_quality": working_row.get("setup_quality_label", "MIXED"),
        "signal_explainer": build_signal_explainer(
            signal=signal,
            signal_strength=signal_strength,
            market_state_label=market_state_label,
        ),
        "confidence_explainer": build_confidence_explainer(),
        "ai_summary": ai_summary,
        "bullish_factors": deduplicate_items(bullish_factors),
        "bearish_factors": deduplicate_items(bearish_factors),
        "watchpoints": deduplicate_items(watchpoints),
    }

    return decision_snapshot


if __name__ == "__main__":
    print("decision_engine.py loaded successfully.")