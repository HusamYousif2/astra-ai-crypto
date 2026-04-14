import os

import joblib

from .engine.pipeline_v2 import build_live_snapshot_v2_for_coin
from .lstm_model import get_model_paths
from .models_ai.registry import infer_direction_v2, infer_setup_quality_v2
from .predict import (
    calculate_practical_risk_metrics,
    lstm_forecast,
    prepare_data,
    run_fast_backtest,
)


LEGACY_FEATURE_COLUMNS = [
    "close",
    "rsi",
    "macd",
    "macd_signal",
    "sma_20",
    "ema_20",
    "volume_mean",
    "atr",
]


def load_legacy_forecast_model(coin):
    """
    Load the legacy sequence model if it exists.
    This is used only as a temporary forecast source while V2 evolves.
    """
    model_path, scaler_path = get_model_paths(coin)

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        return None, None

    from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

    model = keras_load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def build_forecast_bundle(df, coin):
    """
    Build a temporary forecast bundle using the current legacy sequence model.
    """
    model, scaler = load_legacy_forecast_model(coin)

    if model is None or scaler is None:
        return {
            "forecast_values": [],
            "backtest": {"win_rate": 0, "pnl_pct": 0, "trades": 0},
            "advanced_risk": {
                "max_drawdown_pct": 0.0,
                "var_95_pct": 0.0,
                "risk_reward_ratio": None,
            },
        }

    coin_df = prepare_data(df, coin)
    data_matrix = coin_df[LEGACY_FEATURE_COLUMNS].values

    forecast_values = lstm_forecast(model, scaler, data_matrix)

    backtest = run_fast_backtest(model, scaler, data_matrix)

    recent_historical = coin_df["close"].values[-24:]
    advanced_risk = calculate_practical_risk_metrics(
        historical_prices=recent_historical,
        predicted_prices=forecast_values,
    )

    return {
        "forecast_values": forecast_values,
        "backtest": backtest,
        "advanced_risk": advanced_risk,
    }


def build_scenario_analysis(current_price, forecast_context, technical_context_v2, advanced_risk):
    """
    Build cleaner UI-friendly scenario output.
    """
    if not forecast_context:
        return {
            "bull_case": 0.0,
            "base_case": 0.0,
            "bear_case": 0.0,
        }

    forecast_max = forecast_context.get("forecast_max", current_price)
    forecast_min = forecast_context.get("forecast_min", current_price)
    forecast_last = forecast_context.get("forecast_last", current_price)

    raw_bull_case = ((forecast_max - current_price) / current_price) * 100.0
    raw_base_case = ((forecast_last - current_price) / current_price) * 100.0
    raw_bear_case = ((forecast_min - current_price) / current_price) * 100.0

    atr_pct = technical_context_v2.get("atr_pct") or 0.0
    var_95_pct = advanced_risk.get("var_95_pct", 0.0) or 0.0

    atr_stress_case = -(atr_pct * 100.0 * 1.25) if atr_pct else 0.0
    var_stress_case = -abs(var_95_pct)
    minimum_stress_case = -0.10

    bull_case = max(raw_bull_case, 0.0)
    base_case = raw_base_case
    bear_case = min(raw_bear_case, atr_stress_case, var_stress_case, minimum_stress_case)

    return {
        "bull_case": round(float(bull_case), 2),
        "base_case": round(float(base_case), 2),
        "bear_case": round(float(bear_case), 2),
    }


def normalize_technical_context(technical_context_v2):
    """
    Convert V2 technical keys into a UI-friendly structure.
    """
    return {
        "rsi": technical_context_v2.get("rsi"),
        "macd": technical_context_v2.get("macd"),
        "macd_signal": technical_context_v2.get("macd_signal"),
        "ema20": technical_context_v2.get("ema_20"),
        "sma20": technical_context_v2.get("sma_20"),
        "atr": technical_context_v2.get("atr"),
        "atr_pct": technical_context_v2.get("atr_pct"),
        "ret_1": technical_context_v2.get("ret_1"),
        "ret_6": technical_context_v2.get("ret_6"),
        "ret_24": technical_context_v2.get("ret_24"),
        "realized_vol_24": technical_context_v2.get("realized_vol_24"),
        "volume_vs_ma24": technical_context_v2.get("volume_vs_ma24"),
        "range_pct": technical_context_v2.get("range_pct"),
        "trend_alignment_flag": technical_context_v2.get("trend_alignment_flag"),
        "compression_flag": technical_context_v2.get("compression_flag"),
        "expansion_flag": technical_context_v2.get("expansion_flag"),
    }


def map_risk_level_for_ui(risk_level):
    """
    Convert V2 risk level into a UI-friendly posture.
    """
    if not risk_level:
        return "MEDIUM"
    return str(risk_level).upper()


def map_direction_from_signal(signal):
    """
    Ensure we always return a clean direction field.
    """
    if signal == "BUY":
        return "UP"
    if signal == "SELL":
        return "DOWN"
    return "NEUTRAL"


def normalize_risk_reward(risk_reward_ratio, trade_stance):
    """
    Clean up unreliable or visually misleading risk/reward values.
    """
    if risk_reward_ratio is None:
        return None

    try:
        value = float(risk_reward_ratio)
    except Exception:
        return None

    if trade_stance == "No Trade":
        return None

    if value <= 0:
        return None

    if value >= 8.0:
        return None

    return round(value, 2)


def derive_mtf_view(decision, market_state):
    """
    Build a cleaner multi-timeframe view.
    """
    signal = decision.get("signal", "HOLD")
    trade_stance = decision.get("trade_stance", "No Trade")
    trend_state = market_state.get("trend_state", "MIXED")
    momentum_state = market_state.get("momentum_state", "MIXED")
    market_regime = market_state.get("market_state_label", "Transitional")

    if signal == "HOLD" or trade_stance == "No Trade":
        return {
            "mtf_short": "MIXED",
            "mtf_med": "MIXED",
            "mtf_long": "MIXED",
        }

    if momentum_state == "POSITIVE":
        mtf_short = "UP"
    elif momentum_state == "NEGATIVE":
        mtf_short = "DOWN"
    else:
        mtf_short = "MIXED"

    if trend_state == "UPTREND":
        mtf_med = "UP"
    elif trend_state == "DOWNTREND":
        mtf_med = "DOWN"
    else:
        mtf_med = "MIXED"

    if market_regime == "Bullish Momentum":
        mtf_long = "UP"
    elif market_regime == "Bearish Momentum":
        mtf_long = "DOWN"
    else:
        mtf_long = "MIXED"

    return {
        "mtf_short": mtf_short,
        "mtf_med": mtf_med,
        "mtf_long": mtf_long,
    }


def build_display_news_summary(news_context_v2, coin):
    """
    Clean up no-news wording to make it look more professional.
    """
    article_count = news_context_v2.get("article_count", 0)
    summary = news_context_v2.get("summary", "") or ""

    if article_count == 0:
        return f"No strong asset-specific news catalyst is active for {coin} in the current cycle."

    if "No meaningful recent news was detected" in summary:
        return f"No strong asset-specific news catalyst is active for {coin} in the current cycle."

    return summary


def sanitize_decision_snapshot(decision_snapshot):
    """
    Remove internal inconsistencies before sending the payload to the UI.
    """
    clean = dict(decision_snapshot)

    signal = clean.get("signal", "HOLD")
    if signal == "HOLD":
        clean["direction"] = "NEUTRAL"

    return clean


def build_model_context(df, coin):
    """
    Run the trained V2 models and return their outputs.
    """
    coin_df = df[df["symbol"] == coin].copy().sort_values("time")

    try:
        direction_model = infer_direction_v2(coin_df)
    except Exception:
        direction_model = None

    try:
        setup_model = infer_setup_quality_v2(coin_df)
    except Exception:
        setup_model = None

    return {
        "direction_model": direction_model,
        "setup_model": setup_model,
    }


def apply_model_overrides(decision, model_context):
    """
    Blend trained model outputs into the existing V2 decision snapshot.
    The trained models contribute to the decision but do not fully replace it.
    """
    clean = dict(decision)

    direction_model = model_context.get("direction_model")
    setup_model = model_context.get("setup_model")

    original_setup_score = float(clean.get("setup_score", 0.0))
    original_signal_strength = clean.get("signal_strength", "Weak")
    original_signal = clean.get("signal", "HOLD")

    model_notes = []

    if direction_model:
        direction_label = direction_model.get("label", "NEUTRAL")
        direction_score = float(direction_model.get("score", 0.0))
        direction_strength = direction_model.get("strength", "Weak")

        model_notes.append(
            f"Direction model reads {direction_label.lower()} with {direction_strength.lower()} strength."
        )

        if direction_label == "NEUTRAL" and direction_score >= 65:
            clean["signal"] = "HOLD"
            clean["trade_stance"] = "No Trade"

        elif direction_label == "UP" and direction_score >= 72 and original_signal != "SELL":
            if clean.get("signal") == "HOLD":
                clean["signal"] = "BUY"

        elif direction_label == "DOWN" and direction_score >= 72 and original_signal != "BUY":
            if clean.get("signal") == "HOLD":
                clean["signal"] = "SELL"

        if direction_strength == "Strong":
            if clean.get("signal_strength") == "Weak":
                clean["signal_strength"] = "Moderate"

    if setup_model:
        setup_label = setup_model.get("label", "MIXED")
        setup_score = float(setup_model.get("score", 0.0))
        setup_strength = setup_model.get("strength", "Weak")

        clean["setup_quality"] = setup_label

        model_notes.append(
            f"Setup model reads {setup_label.lower()} structure with {setup_strength.lower()} confidence."
        )

        blended_setup_score = (original_setup_score * 0.65) + (setup_score * 0.35)
        clean["setup_score"] = round(float(blended_setup_score), 1)

        if setup_label == "WEAK":
            clean["trade_stance"] = "No Trade"
            if clean.get("signal") in {"BUY", "SELL"}:
                clean["signal"] = "HOLD"

        elif setup_label == "CLEAN" and setup_score >= 60:
            if clean.get("signal") != "HOLD" and clean.get("trade_stance") != "No Trade":
                clean["trade_stance"] = "Trade Candidate"

    if clean.get("signal") == "HOLD":
        clean["direction"] = "NEUTRAL"
        clean["trade_stance"] = "No Trade"

    if model_notes:
        watchpoints = list(clean.get("watchpoints", []))
        for note in model_notes:
            if note not in watchpoints:
                watchpoints.append(note)
        clean["watchpoints"] = watchpoints[:4]

    return clean


def predict_coin_v2(df, coin):
    """
    Build a V2 intelligence payload while preserving compatibility
    with the current dashboard and storage structure.
    """
    forecast_bundle = build_forecast_bundle(df, coin)
    forecast_values = forecast_bundle["forecast_values"]

    snapshot_v2 = build_live_snapshot_v2_for_coin(
        df=df,
        coin=coin,
        forecast_values=forecast_values,
    )

    decision = sanitize_decision_snapshot(snapshot_v2.get("decision_snapshot", {}))
    model_context = build_model_context(df, coin)
    decision = apply_model_overrides(decision, model_context)

    market_state = snapshot_v2.get("market_state", {})
    risk_snapshot = snapshot_v2.get("risk_snapshot", {})
    technical_context_v2 = snapshot_v2.get("technical_context_v2", {})
    news_context_v2 = snapshot_v2.get("news_context", {})

    current_price = float(snapshot_v2["current_price"])

    scenario_analysis = build_scenario_analysis(
        current_price=current_price,
        forecast_context=snapshot_v2.get("forecast_context", {}),
        technical_context_v2=technical_context_v2,
        advanced_risk=forecast_bundle["advanced_risk"],
    )

    technical_context = normalize_technical_context(technical_context_v2)

    direction = map_direction_from_signal(decision.get("signal", "HOLD"))
    risk_posture = map_risk_level_for_ui(risk_snapshot.get("risk_level", "Medium"))

    mtf_view = derive_mtf_view(
        decision=decision,
        market_state=market_state,
    )

    news_context = {
        "sentiment_score": news_context_v2.get("sentiment_score", 0.0),
        "sentiment_label": news_context_v2.get("sentiment_label", "NEUTRAL"),
        "article_count": news_context_v2.get("article_count", 0),
        "summary": build_display_news_summary(news_context_v2, coin),
        "top_articles": news_context_v2.get("top_articles", []),
        "news_impact": news_context_v2.get("news_impact", "Low"),
        "news_relevance": news_context_v2.get("news_relevance", "Low"),
        "contradiction_with_market": news_context_v2.get("contradiction_with_market", False),
        "top_drivers": news_context_v2.get("top_drivers", []),
    }

    advanced_risk = forecast_bundle["advanced_risk"]
    backtest = forecast_bundle["backtest"]

    volatility_value = technical_context_v2.get("realized_vol_24", 0.0)
    trend_strength_value = market_state.get("market_state_score", 0.0)

    clean_risk_reward = normalize_risk_reward(
        risk_reward_ratio=advanced_risk.get("risk_reward_ratio"),
        trade_stance=decision.get("trade_stance", "No Trade"),
    )

    payload = {
        # Legacy-compatible fields
        "current_price": current_price,
        "forecast_next_hours": forecast_values,
        "direction": direction,
        "confidence": round(float((decision.get("setup_score", 0.0) / 100.0)), 3),
        "insight": decision.get("ai_summary", ""),
        "volatility": float(volatility_value or 0.0),
        "trend_strength": float(trend_strength_value or 0.0),
        "risk_level": f"{risk_posture} RISK",
        "nlp_score": news_context.get("sentiment_score", 0.0),
        "max_drawdown": float(advanced_risk.get("max_drawdown_pct", 0.0)),
        "value_at_risk": float(advanced_risk.get("var_95_pct", 0.0)),
        "risk_reward": clean_risk_reward,
        "chart_data": snapshot_v2.get("chart_data", []),
        "mtf_short": mtf_view["mtf_short"],
        "mtf_med": mtf_view["mtf_med"],
        "mtf_long": mtf_view["mtf_long"],
        "bt_win_rate": backtest.get("win_rate", 0),
        "bt_pnl": backtest.get("pnl_pct", 0),
        "bt_trades": backtest.get("trades", 0),

        # V2 fields
        "signal": decision.get("signal", "HOLD"),
        "signal_strength": decision.get("signal_strength", "Weak"),
        "trade_stance": decision.get("trade_stance", "No Trade"),
        "setup_score": decision.get("setup_score", 0.0),
        "market_regime": market_state.get("market_state_label", "Transitional"),
        "risk_posture": risk_posture,
        "trade_safety": risk_snapshot.get("trade_safety", "Caution"),
        "setup_quality": decision.get("setup_quality", "MIXED"),
        "news_impact": decision.get("news_impact", "Low"),
        "confidence_explainer": decision.get("confidence_explainer", ""),
        "signal_explainer": decision.get("signal_explainer", ""),
        "ai_summary": decision.get("ai_summary", ""),
        "bullish_factors": decision.get("bullish_factors", []),
        "bearish_factors": decision.get("bearish_factors", []),
        "watchpoints": decision.get("watchpoints", []),
        "scenario_analysis": scenario_analysis,
        "news_context": news_context,
        "technical_context": technical_context,

        # Full V2 snapshots for later use
        "market_state_snapshot": market_state,
        "risk_snapshot": risk_snapshot,
        "decision_snapshot": decision,
        "technical_context_v2": technical_context_v2,
        "forecast_context": snapshot_v2.get("forecast_context", {}),
        "model_context": model_context,
        "v2_timestamp": snapshot_v2.get("timestamp"),
    }

    return payload


if __name__ == "__main__":
    print("predict_v2.py loaded successfully.")