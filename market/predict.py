import os
import logging

# Suppress TensorFlow backend warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("absl").setLevel(logging.ERROR)

import joblib
import numpy as np
from .features import add_indicators
from .lstm_model import SEQ_LENGTH, create_sequences, get_model_paths
from .nlp_engine import analyze_coin_sentiment


def prepare_data(df, coin):
    """
    Filter data for a specific coin and add technical indicators.
    """
    coin_df = df[df["symbol"] == coin].copy()
    coin_df = add_indicators(coin_df)
    coin_df = coin_df.dropna().reset_index(drop=True)
    return coin_df


def run_fast_backtest(model, scaler, data_matrix, requested_lookback=168):
    """
    Run a lightweight historical simulation over the most recent window.
    """
    max_available = len(data_matrix) - SEQ_LENGTH - 1

    if max_available < 10:
        return {"win_rate": 0, "pnl_pct": 0, "trades": 0}

    actual_lookback = min(requested_lookback, max_available)
    test_data = data_matrix[-(actual_lookback + SEQ_LENGTH):]
    scaled_test_data = scaler.transform(test_data)

    X_batch, y_batch_scaled = create_sequences(scaled_test_data, SEQ_LENGTH)

    if len(X_batch) == 0:
        return {"win_rate": 0, "pnl_pct": 0, "trades": 0}

    scaled_preds = model.predict(X_batch, verbose=0)

    dummy_preds = np.zeros((len(scaled_preds), data_matrix.shape[1]))
    dummy_preds[:, 0] = scaled_preds[:, 0]
    actual_preds = scaler.inverse_transform(dummy_preds)[:, 0]

    dummy_y = np.zeros((len(y_batch_scaled), data_matrix.shape[1]))
    dummy_y[:, 0] = y_batch_scaled
    actual_y = scaler.inverse_transform(dummy_y)[:, 0]

    dummy_current = np.zeros((len(X_batch), data_matrix.shape[1]))
    dummy_current[:, 0] = X_batch[:, -1, 0]
    actual_current = scaler.inverse_transform(dummy_current)[:, 0]

    winning_trades = 0
    total_trades = 0
    portfolio_value = 10000.0

    for i in range(len(actual_preds)):
        pred_dir = 1 if actual_preds[i] > actual_current[i] else -1
        actual_dir = 1 if actual_y[i] > actual_current[i] else -1

        price_diff_pct = abs(actual_preds[i] - actual_current[i]) / actual_current[i]

        if price_diff_pct > 0.001:
            total_trades += 1

            if pred_dir == actual_dir:
                winning_trades += 1

            hourly_return = (actual_y[i] - actual_current[i]) / actual_current[i]
            if pred_dir == 1:
                portfolio_value *= (1 + hourly_return)
            else:
                portfolio_value *= (1 - hourly_return)

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    pnl_pct = ((portfolio_value - 10000) / 10000) * 100

    return {
        "win_rate": round(win_rate, 1),
        "pnl_pct": round(pnl_pct, 2),
        "trades": total_trades,
    }


def lstm_forecast(model, scaler, data_matrix, steps=6):
    """
    Generate a multi-step forecast sequentially.
    """
    preds = []
    current_seq = data_matrix[-SEQ_LENGTH:]

    for _ in range(steps):
        current_seq_scaled = scaler.transform(current_seq)
        current_seq_3d = np.expand_dims(current_seq_scaled, axis=0)

        pred_scaled = model.predict(current_seq_3d, verbose=0)[0][0]

        dummy = np.zeros((1, current_seq.shape[1]))
        dummy[0, 0] = pred_scaled
        pred_price = scaler.inverse_transform(dummy)[0, 0]

        preds.append(float(pred_price))

        new_row = current_seq[-1].copy()
        new_row[0] = pred_price
        current_seq = np.vstack([current_seq[1:], new_row])

    return preds


def calculate_practical_risk_metrics(historical_prices, predicted_prices):
    """
    Calculate practical risk metrics for dashboard and decision support.
    """
    peak = np.max(historical_prices)
    trough = np.min(historical_prices)
    max_drawdown = (peak - trough) / peak if peak > 0 else 0

    returns = np.diff(historical_prices) / historical_prices[:-1]
    var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0.0

    current_price = historical_prices[-1]
    expected_high = np.max(predicted_prices)
    expected_low = np.min(predicted_prices)

    potential_reward = (expected_high - current_price) / current_price
    potential_risk = (current_price - expected_low) / current_price

    risk_reward_ratio = 99.9 if potential_risk <= 0.0001 else potential_reward / potential_risk

    return {
        "max_drawdown_pct": round(float(max_drawdown * 100), 2),
        "var_95_pct": round(float(abs(var_95) * 100), 2),
        "risk_reward_ratio": round(float(risk_reward_ratio), 2),
    }


def classify_market_regime(current_price, sma20, ema20, rsi, macd, macd_signal, volatility):
    """
    Classify the current market state into a human-readable regime.
    """
    if current_price > ema20 > sma20 and macd > macd_signal and 45 <= rsi <= 70:
        return "Bullish Momentum"

    if current_price < ema20 < sma20 and macd < macd_signal and 30 <= rsi <= 55:
        return "Bearish Momentum"

    if volatility > 0.02:
        return "High Volatility"

    if 45 <= rsi <= 55 and abs(macd - macd_signal) < 10:
        return "Range-Bound"

    return "Transitional"


def build_professional_signal(current_price, preds, nlp_result, indicators, advanced_risk):
    """
    Build a more professional decision layer with clear rationale.
    """
    target_price = preds[-1]
    expected_move_pct = ((target_price - current_price) / current_price) * 100

    rsi = indicators["rsi"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    ema20 = indicators["ema20"]
    sma20 = indicators["sma20"]
    sentiment_score = nlp_result["score"]

    bullish_factors = []
    bearish_factors = []
    watchpoints = []

    score = 0.0

    # Forecast contribution
    if expected_move_pct > 1.0:
        score += 2
        bullish_factors.append("Model forecast points to meaningful upside over the next few hours.")
    elif expected_move_pct > 0.2:
        score += 1
        bullish_factors.append("Forecast remains positive, but upside is moderate.")
    elif expected_move_pct < -1.0:
        score -= 2
        bearish_factors.append("Model forecast points to meaningful downside over the next few hours.")
    elif expected_move_pct < -0.2:
        score -= 1
        bearish_factors.append("Forecast remains negative, but downside is moderate.")

    # Trend structure
    if current_price > ema20 > sma20:
        score += 1.5
        bullish_factors.append("Price is trading above key short-term trend averages.")
    elif current_price < ema20 < sma20:
        score -= 1.5
        bearish_factors.append("Price is trading below key short-term trend averages.")

    # Momentum
    if macd > macd_signal:
        score += 1
        bullish_factors.append("MACD momentum remains supportive.")
    else:
        score -= 1
        bearish_factors.append("MACD momentum remains weak.")

    # RSI context
    if 50 <= rsi <= 68:
        score += 0.8
        bullish_factors.append("RSI shows constructive momentum without being overheated.")
    elif 32 <= rsi < 50:
        score -= 0.6
        bearish_factors.append("RSI remains below neutral momentum territory.")
    elif rsi > 72:
        bearish_factors.append("RSI is elevated and raises short-term reversal risk.")
        watchpoints.append("Watch for momentum exhaustion after an overbought move.")
    elif rsi < 28:
        bullish_factors.append("RSI is deeply stretched and may support a relief rebound.")
        watchpoints.append("Watch for confirmation of a rebound from oversold conditions.")

    # Sentiment
    if sentiment_score > 0.20:
        score += 1
        bullish_factors.append("News sentiment is supportive.")
    elif sentiment_score < -0.20:
        score -= 1
        bearish_factors.append("News sentiment is negative.")
    else:
        watchpoints.append("News flow is not yet a strong directional driver.")

    # Risk
    if advanced_risk["var_95_pct"] > 3.0:
        bearish_factors.append("Short-term downside risk remains elevated.")
        watchpoints.append("Position sizing should remain conservative.")
    if advanced_risk["risk_reward_ratio"] < 1.0:
        bearish_factors.append("Risk/reward is not compelling at current levels.")

    # Signal mapping
    if score >= 3:
        signal = "BUY"
        direction = "UP"
    elif score <= -3:
        signal = "SELL"
        direction = "DOWN"
    else:
        signal = "HOLD"
        direction = "UP" if expected_move_pct >= 0 else "DOWN"

    confidence = min(max(abs(score) / 6, 0.20), 0.95)

    if advanced_risk["var_95_pct"] > 5.0 or advanced_risk["max_drawdown_pct"] > 10.0:
        risk_posture = "HIGH"
    elif advanced_risk["var_95_pct"] > 2.0:
        risk_posture = "MEDIUM"
    else:
        risk_posture = "LOW"

    if signal == "BUY":
        ai_summary = (
            "Market structure is constructive. The model, momentum, and trend context support a bullish bias, "
            "but execution should still respect current risk conditions."
        )
    elif signal == "SELL":
        ai_summary = (
            "Market structure is fragile. The model and momentum context favor downside risk, "
            "so defensive positioning is more appropriate at this stage."
        )
    else:
        ai_summary = (
            "The market does not show a clean directional edge right now. The setup looks mixed, "
            "so patience is more appropriate than aggressive positioning."
        )

    if not bullish_factors:
        bullish_factors.append("No strong bullish driver is currently dominant.")

    if not bearish_factors:
        bearish_factors.append("No major bearish driver is currently dominant.")

    if not watchpoints:
        watchpoints.append("Monitor whether price confirms the forecast path with stronger volume and momentum.")

    scenario_analysis = {
        "bull_case": round(float((max(preds) - current_price) / current_price * 100), 2),
        "base_case": round(float((preds[-1] - current_price) / current_price * 100), 2),
        "bear_case": round(float((min(preds) - current_price) / current_price * 100), 2),
    }

    return {
        "signal": signal,
        "direction": direction,
        "confidence": round(float(confidence), 3),
        "risk_posture": risk_posture,
        "ai_summary": ai_summary,
        "bullish_factors": bullish_factors[:4],
        "bearish_factors": bearish_factors[:4],
        "watchpoints": watchpoints[:4],
        "scenario_analysis": scenario_analysis,
    }


def predict_coin(df, coin):
    """
    Main orchestrator for institutional-style market intelligence output.
    """
    features_list = [
        "close",
        "rsi",
        "macd",
        "macd_signal",
        "sma_20",
        "ema_20",
        "volume_mean",
        "atr",
    ]

    coin_df = prepare_data(df, coin)
    data_matrix = coin_df[features_list].values
    current_price = float(coin_df["close"].iloc[-1])

    model_path, scaler_path = get_model_paths(coin)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

        model = keras_load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        raise Exception(f"AI model for {coin} is currently training in the background. Please wait.")

    preds = lstm_forecast(model, scaler, data_matrix)

    try:
        nlp_result = analyze_coin_sentiment(coin)
    except Exception:
        nlp_result = {"score": 0.0, "label": "NEUTRAL", "article_count": 0}

    current_indicators = {
        "rsi": float(coin_df["rsi"].iloc[-1]),
        "macd": float(coin_df["macd"].iloc[-1]),
        "macd_signal": float(coin_df["macd_signal"].iloc[-1]),
        "ema20": float(coin_df["ema_20"].iloc[-1]),
        "sma20": float(coin_df["sma_20"].iloc[-1]),
        "atr": float(coin_df["atr"].iloc[-1]),
    }

    backtest_results = run_fast_backtest(model, scaler, data_matrix)

    returns = np.diff(preds) / preds[:-1]
    volatility = float(np.std(returns))

    recent_historical = coin_df["close"].values[-24:]
    advanced_risk = calculate_practical_risk_metrics(recent_historical, preds)

    professional_decision = build_professional_signal(
        current_price=current_price,
        preds=preds,
        nlp_result=nlp_result,
        indicators=current_indicators,
        advanced_risk=advanced_risk,
    )

    market_regime = classify_market_regime(
        current_price=current_price,
        sma20=current_indicators["sma20"],
        ema20=current_indicators["ema20"],
        rsi=current_indicators["rsi"],
        macd=current_indicators["macd"],
        macd_signal=current_indicators["macd_signal"],
        volatility=volatility,
    )

    if advanced_risk["var_95_pct"] > 5.0 or advanced_risk["max_drawdown_pct"] > 10.0:
        risk_level = "HIGH RISK"
    elif advanced_risk["var_95_pct"] > 2.0:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    recent_candles = []
    for _, row in coin_df.tail(100).iterrows():
        recent_candles.append(
            {
                "time": int(row["time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "value": float(row["volumeto"]),
            }
        )

    technical_context = {
        "rsi": round(current_indicators["rsi"], 2),
        "macd": round(current_indicators["macd"], 2),
        "macd_signal": round(current_indicators["macd_signal"], 2),
        "ema20": round(current_indicators["ema20"], 2),
        "sma20": round(current_indicators["sma20"], 2),
        "atr": round(current_indicators["atr"], 2),
    }

    news_context = {
    "sentiment_score": nlp_result["score"],
    "sentiment_label": nlp_result["label"],
    "article_count": nlp_result["article_count"],
    "summary": nlp_result.get("summary", ""),
    "top_articles": nlp_result.get("articles", []),
    }

    return {
        # Legacy fields kept for current dashboard compatibility
        "current_price": current_price,
        "forecast_next_hours": preds,
        "direction": professional_decision["direction"],
        "confidence": professional_decision["confidence"],
        "insight": professional_decision["ai_summary"],
        "volatility": volatility,
        "trend_strength": float(np.mean(returns)),
        "risk_level": risk_level,
        "nlp_score": nlp_result["score"],
        "max_drawdown": advanced_risk["max_drawdown_pct"],
        "value_at_risk": advanced_risk["var_95_pct"],
        "risk_reward": advanced_risk["risk_reward_ratio"],
        "chart_data": recent_candles,
        "mtf_short": professional_decision["direction"],
        "mtf_med": "UP" if current_price > current_indicators["ema20"] else "DOWN",
        "mtf_long": "UP" if current_price > current_indicators["sma20"] else "DOWN",
        "bt_win_rate": backtest_results["win_rate"],
        "bt_pnl": backtest_results["pnl_pct"],
        "bt_trades": backtest_results["trades"],

        # New professional fields
        "signal": professional_decision["signal"],
        "market_regime": market_regime,
        "risk_posture": professional_decision["risk_posture"],
        "ai_summary": professional_decision["ai_summary"],
        "bullish_factors": professional_decision["bullish_factors"],
        "bearish_factors": professional_decision["bearish_factors"],
        "watchpoints": professional_decision["watchpoints"],
        "scenario_analysis": professional_decision["scenario_analysis"],
        "news_context": news_context,
        "technical_context": technical_context,
    }