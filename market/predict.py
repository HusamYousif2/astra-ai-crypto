# import os
# import logging

# # 1. Suppress core TensorFlow warnings (0=all, 1=info, 2=warnings, 3=errors)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # 2. Suppress 'absl' specific warnings (like the compilation warning)
# logging.getLogger('absl').setLevel(logging.ERROR)
# import numpy as np
# import pandas as pd
# import os
# from tensorflow.keras.models import load_model as keras_load_model # type: ignore
# import joblib

# from .features import add_indicators
# from .lstm_model import get_model_paths, create_sequences, train_lstm_for_coin, SEQ_LENGTH
# from .nlp_engine import analyze_coin_sentiment

# def prepare_data(df, coin):
#     """
#     Filter data for the specific coin and apply technical indicators.
#     """
#     coin_df = df[df["symbol"] == coin].copy()
#     coin_df = add_indicators(coin_df)
#     coin_df = coin_df.dropna()
#     return coin_df


# def lstm_forecast(model, scaler, data_matrix, steps=6):
#     """
#     Generate future price path for N steps sequentially.
#     """
#     preds = []
#     # Get the last known window of data
#     current_seq = data_matrix[-SEQ_LENGTH:]
    
#     for _ in range(steps):
#         current_seq_scaled = scaler.transform(current_seq)
#         current_seq_3d = np.expand_dims(current_seq_scaled, axis=0)
        
#         # Predict the next scaled price (feature index 0)
#         pred_scaled = model.predict(current_seq_3d, verbose=0)[0][0]
        
#         # Inverse transform to get actual USD price
#         dummy = np.zeros((1, current_seq.shape[1]))
#         dummy[0, 0] = pred_scaled
#         pred_price = scaler.inverse_transform(dummy)[0, 0]
        
#         preds.append(float(pred_price))
        
#         # Shift sequence forward: drop oldest, append new predicted row
#         new_row = current_seq[-1].copy()
#         new_row[0] = pred_price 
#         current_seq = np.vstack([current_seq[1:], new_row])
        
#     return preds

# def ensemble_decision(current_price, lstm_preds, nlp_sentiment):
#     """
#     Merge the Quantitative DL model with Qualitative NLP model.
#     This is the core algorithm used by hedge funds to manage risk.
#     """
#     lstm_target = lstm_preds[-1]
#     price_change_pct = (lstm_target - current_price) / current_price
    
#     lstm_direction = "UP" if price_change_pct > 0 else "DOWN"
#     # Base technical confidence caps at 80%
#     lstm_confidence = min(abs(price_change_pct) * 15, 0.8) 
    
#     nlp_score = nlp_sentiment['score']
    
#     final_confidence = lstm_confidence
#     final_direction = lstm_direction
#     insight_msg = ""
    
#     # Logic matrix: Convergences and Divergences
#     if lstm_direction == "UP" and nlp_score > 0.1:
#         final_confidence += nlp_score * 0.2
#         insight_msg = "Strong Bullish Convergence: Technical momentum and News sentiment are both aligned."
#     elif lstm_direction == "DOWN" and nlp_score < -0.1:
#         final_confidence += abs(nlp_score) * 0.2
#         insight_msg = "Strong Bearish Convergence: Technical breakdown confirmed by negative News sentiment."
#     elif lstm_direction == "UP" and nlp_score < -0.2:
#         final_confidence -= abs(nlp_score) * 0.3
#         insight_msg = "Warning Divergence: Technicals show upward momentum, but News sentiment is highly bearish. High risk of reversal."
#         if final_confidence < 0.3: final_direction = "NEUTRAL"
#     elif lstm_direction == "DOWN" and nlp_score > 0.2:
#         final_confidence -= nlp_score * 0.3
#         insight_msg = "Warning Divergence: Technical breakdown detected, but positive News might cushion the fall."
#         if final_confidence < 0.3: final_direction = "NEUTRAL"
#     else:
#         insight_msg = "Market is driven primarily by technical indicators. News impact is currently minimal."
        
#     return {
#         "direction": final_direction,
#         "confidence": round(float(max(0.1, min(final_confidence, 0.99))), 3),
#         "insight": insight_msg
#     }


# def calculate_practical_risk_metrics(historical_prices, predicted_prices):
#     """
#     Calculate practical business-oriented risk metrics for regional investors.
#     """
#     # 1. Maximum Drawdown (MDD) - "How bad can the drop be?"
#     # Calculated on the last 24 hours of data
#     peak = np.max(historical_prices)
#     trough = np.min(historical_prices)
#     max_drawdown = (peak - trough) / peak if peak > 0 else 0
    
#     # 2. Value at Risk (VaR 95%) - "What is the worst expected loss in a normal day?"
#     # Using historical simulation method
#     returns = np.diff(historical_prices) / historical_prices[:-1]
#     if len(returns) > 0:
#         var_95 = np.percentile(returns, 5) # The 5th percentile worst return
#     else:
#         var_95 = 0.0
        
#     # 3. Risk/Reward Ratio based on AI Forecast
#     current_price = historical_prices[-1]
#     expected_high = np.max(predicted_prices)
#     expected_low = np.min(predicted_prices)
    
#     potential_reward = (expected_high - current_price) / current_price
#     potential_risk = (current_price - expected_low) / current_price
    
#     # Avoid division by zero
#     if potential_risk <= 0.0001:
#         risk_reward_ratio = 99.9 # Highly favorable
#     else:
#         risk_reward_ratio = potential_reward / potential_risk

#     return {
#         "max_drawdown_pct": round(float(max_drawdown * 100), 2),
#         "var_95_pct": round(float(abs(var_95) * 100), 2), 
#         "risk_reward_ratio": round(float(risk_reward_ratio), 2)
#     }


# # def predict_coin(df, coin):
# #     """
# #     The main orchestrator called by Django views.
# #     """
# #     features_list = [
# #         "close", "rsi", "macd", "macd_signal", 
# #         "sma_20", "ema_20", "volume_mean", "atr"
# #     ]
    
# #     coin_df = prepare_data(df, coin)
# #     data_matrix = coin_df[features_list].values
# #     current_price = float(coin_df["close"].iloc[-1])
    
# #     model_path, scaler_path = get_model_paths(coin)
    
# #     # 1. Load the pre-trained Deep Learning Model (Lightning Fast!)
# #     if os.path.exists(model_path) and os.path.exists(scaler_path):
# #         model = keras_load_model(model_path)
# #         scaler = joblib.load(scaler_path)
# #     else:
# #         # Fallback: if background task hasn't created the model yet
# #         raise Exception(f"AI Model for {coin} is currently training in the background. Please wait a moment.")
        
# #     # 2. Generate Technical Forecast
# #     preds = lstm_forecast(model, scaler, data_matrix)
    
# #     # 3. Fetch Fundamental Sentiment
# #     try:
# #         nlp_result = analyze_coin_sentiment(coin)
# #     except Exception as e:
# #         print(f"NLP Engine error for {coin}: {e}")
# #         nlp_result = {"score": 0.0, "label": "NEUTRAL", "article_count": 0}
        
# #     # 4. Process Ensemble Logic
# #     decision = ensemble_decision(current_price, preds, nlp_result)
    

# #     # 5. Calculate Risk Metrics
# #     returns = np.diff(preds) / preds[:-1]
# #     volatility = float(np.std(returns))
    
# #     # Extract recent historical prices for risk calculations (last 24 hours)
# #     recent_historical = coin_df["close"].values[-24:]
    
# #     # Call the new practical risk engine
# #     advanced_risk = calculate_practical_risk_metrics(recent_historical, preds)
    
# #     # Simplified risk logic for business users
# #     if advanced_risk["var_95_pct"] > 5.0 or advanced_risk["max_drawdown_pct"] > 10.0:
# #         risk_level = "HIGH RISK"
# #     elif advanced_risk["var_95_pct"] > 2.0:
# #         risk_level = "MEDIUM RISK"
# #     else:
# #         risk_level = "LOW RISK"
        
# #     return {
# #         "current_price": current_price,
# #         "forecast_next_hours": preds,
# #         "direction": decision["direction"],
# #         "confidence": decision["confidence"],
# #         "insight": decision["insight"],
# #         "volatility": volatility,
# #         "trend_strength": float(np.mean(returns)),
# #         "risk_level": risk_level,
# #         "nlp_score": nlp_result["score"],
# #         # Add the new metrics to the output
# #         "max_drawdown": advanced_risk["max_drawdown_pct"],
# #         "value_at_risk": advanced_risk["var_95_pct"],
# #         "risk_reward": advanced_risk["risk_reward_ratio"]
# #     }

# def predict_coin(df, coin):
#     """
#     The main orchestrator called by Django views.
#     """
#     features_list = [
#         "close", "rsi", "macd", "macd_signal", 
#         "sma_20", "ema_20", "volume_mean", "atr"
#     ]
    
#     coin_df = prepare_data(df, coin)
#     data_matrix = coin_df[features_list].values
#     current_price = float(coin_df["close"].iloc[-1])
    
#     model_path, scaler_path = get_model_paths(coin)
    
#     # 1. Load the pre-trained Deep Learning Model
#     if os.path.exists(model_path) and os.path.exists(scaler_path):
#         from tensorflow.keras.models import load_model as keras_load_model # type: ignore
#         model = keras_load_model(model_path)
#         scaler = joblib.load(scaler_path)
#     else:
#         raise Exception(f"AI Model for {coin} is currently training in the background. Please wait.")
        
#     # 2. Generate Technical Forecast
#     preds = lstm_forecast(model, scaler, data_matrix)
    
#     # 3. Fetch Fundamental Sentiment
#     try:
#         nlp_result = analyze_coin_sentiment(coin)
#     except Exception as e:
#         print(f"NLP Engine error for {coin}: {e}")
#         nlp_result = {"score": 0.0, "label": "NEUTRAL", "article_count": 0}
        
#     # 4. Process Ensemble Logic
#     decision = ensemble_decision(current_price, preds, nlp_result)
    
#     # 5. Calculate Risk Metrics
#     returns = np.diff(preds) / preds[:-1]
#     volatility = float(np.std(returns))
    
#     recent_historical = coin_df["close"].values[-24:]
#     advanced_risk = calculate_practical_risk_metrics(recent_historical, preds)
    
#     if advanced_risk["var_95_pct"] > 5.0 or advanced_risk["max_drawdown_pct"] > 10.0:
#         risk_level = "HIGH RISK"
#     elif advanced_risk["var_95_pct"] > 2.0:
#         risk_level = "MEDIUM RISK"
#     else:
#         risk_level = "LOW RISK"

#     # --- NEW: Extract Candlestick Data for TradingView ---
#     recent_candles = []
#     # Get the last 100 hours for charting
#     for _, row in coin_df.tail(100).iterrows():
#         recent_candles.append({
#             "time": int(row["time"].timestamp()),
#             "open": float(row["open"]),
#             "high": float(row["high"]),
#             "low": float(row["low"]),
#             "close": float(row["close"]),
#             "value": float(row["volumeto"]) # Volume for TradingView Histogram
#         })

#     # --- NEW: Multi-Timeframe Confluence (MTF) ---
#     sma20 = float(coin_df["sma_20"].iloc[-1])
#     ema20 = float(coin_df["ema_20"].iloc[-1])
        
#     return {
#         "current_price": current_price,
#         "forecast_next_hours": preds,
#         "direction": decision["direction"],
#         "confidence": decision["confidence"],
#         "insight": decision["insight"],
#         "volatility": volatility,
#         "trend_strength": float(np.mean(returns)),
#         "risk_level": risk_level,
#         "nlp_score": nlp_result["score"],
#         "max_drawdown": advanced_risk["max_drawdown_pct"],
#         "value_at_risk": advanced_risk["var_95_pct"],
#         "risk_reward": advanced_risk["risk_reward_ratio"],
        
#         # New Pro Metrics
#         "chart_data": recent_candles,
#         "mtf_short": decision["direction"],
#         "mtf_med": "UP" if current_price > ema20 else "DOWN",
#         "mtf_long": "UP" if current_price > sma20 else "DOWN"
#     }
# market/predict.py

import numpy as np
import pandas as pd
import os
import joblib

from .features import add_indicators
from .lstm_model import get_model_paths, create_sequences, train_lstm_for_coin, SEQ_LENGTH
from .nlp_engine import analyze_coin_sentiment

def prepare_data(df, coin):
    """Filter data for the specific coin and apply technical indicators."""
    coin_df = df[df["symbol"] == coin].copy()
    coin_df = add_indicators(coin_df)
    coin_df = coin_df.dropna()
    return coin_df

def run_fast_backtest(model, scaler, data_matrix, requested_lookback=168):
    """
    Run a real, honest vectorized backtest.
    Dynamically adjusts the lookback period based on available API data.
    """
    # 1. Dynamic Lookback Adjustment
    # Calculate how many hours we can actually test without crashing
    max_available = len(data_matrix) - SEQ_LENGTH - 1
    
    # If API gave us too little data, return 0 to be safe
    if max_available < 10: 
        return {"win_rate": 0, "pnl_pct": 0, "trades": 0}
        
    # Use 168 (7 days) if possible, otherwise use whatever the API provided
    actual_lookback = min(requested_lookback, max_available)

    # Extract the historical window for testing
    test_data = data_matrix[-(actual_lookback + SEQ_LENGTH):]
    scaled_test_data = scaler.transform(test_data)

    # Create sequences
    X_batch, y_batch_scaled = create_sequences(scaled_test_data, SEQ_LENGTH)

    if len(X_batch) == 0:
        return {"win_rate": 0, "pnl_pct": 0, "trades": 0}

    # Batch prediction (Extremely fast, milliseconds)
    scaled_preds = model.predict(X_batch, verbose=0)

    # Prepare dummy arrays to reverse the scaling and get real USD prices
    dummy_preds = np.zeros((len(scaled_preds), data_matrix.shape[1]))
    dummy_preds[:, 0] = scaled_preds[:, 0]
    actual_preds = scaler.inverse_transform(dummy_preds)[:, 0]

    dummy_y = np.zeros((len(y_batch_scaled), data_matrix.shape[1]))
    dummy_y[:, 0] = y_batch_scaled
    actual_y = scaler.inverse_transform(dummy_y)[:, 0]

    dummy_current = np.zeros((len(X_batch), data_matrix.shape[1]))
    dummy_current[:, 0] = X_batch[:, -1, 0]
    actual_current = scaler.inverse_transform(dummy_current)[:, 0]

    # Backtest simulation variables
    winning_trades = 0
    total_trades = 0
    portfolio_value = 10000.0  # Starting capital ($10k)

    for i in range(len(actual_preds)):
        pred_dir = 1 if actual_preds[i] > actual_current[i] else -1
        actual_dir = 1 if actual_y[i] > actual_current[i] else -1

        # Only execute trade if AI expects a move larger than 0.1% (filtering noise)
        price_diff_pct = abs(actual_preds[i] - actual_current[i]) / actual_current[i]
        
        if price_diff_pct > 0.001:
            total_trades += 1
            if pred_dir == actual_dir:
                winning_trades += 1

            # Simulate PnL (Capturing the actual 1-hour move)
            hourly_return = (actual_y[i] - actual_current[i]) / actual_current[i]
            if pred_dir == 1:
                portfolio_value *= (1 + hourly_return) # Long Trade
            else:
                portfolio_value *= (1 - hourly_return) # Short Trade

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    pnl_pct = ((portfolio_value - 10000) / 10000) * 100

    return {
        "win_rate": round(win_rate, 1),
        "pnl_pct": round(pnl_pct, 2),
        "trades": total_trades
    }

def lstm_forecast(model, scaler, data_matrix, steps=6):
    """Generate future price path sequentially."""
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

def ensemble_decision(current_price, lstm_preds, nlp_sentiment, indicators):
    """
    Combines DL forecast with Mathematical Confluence 
    using SIMPLE, everyday English.
    """
    lstm_target = lstm_preds[-1]
    price_change_pct = (lstm_target - current_price) / current_price
    lstm_direction = "UP" if price_change_pct > 0 else "DOWN"
    
    base_confidence = min(abs(price_change_pct) * 20, 0.40) 
    confluence_score = 0.0
    insight_parts = []

    rsi, macd, macd_sig, ema20 = indicators['rsi'], indicators['macd'], indicators['macd_signal'], indicators['ema20']
    nlp_score = nlp_sentiment['score']

    # --- Simplified English Insights ---
    if lstm_direction == "UP":
        if rsi > 40 and rsi < 70:
            confluence_score += 0.15 
            insight_parts.append("Price has room to grow.")
        if macd > macd_sig:
            confluence_score += 0.20 
            insight_parts.append("Buying pressure is high.")
        if current_price > ema20:
            confluence_score += 0.15 
            insight_parts.append("Trend is going up.")
        if nlp_score > 0.05:
            confluence_score += 0.10 
            
    elif lstm_direction == "DOWN":
        if rsi < 60 and rsi > 30:
            confluence_score += 0.15 
            insight_parts.append("Price has room to drop.")
        if macd < macd_sig:
            confluence_score += 0.20 
            insight_parts.append("Selling pressure is high.")
        if current_price < ema20:
            confluence_score += 0.15 
            insight_parts.append("Trend is going down.")
        if nlp_score < -0.05:
            confluence_score += 0.10 

    final_confidence = base_confidence + confluence_score
    final_confidence = max(0.15, min(final_confidence, 0.98))
    
    if confluence_score >= 0.4:
        insight_msg = "Strong Signal: AI and indicators agree. " + " ".join(insight_parts)
    elif confluence_score >= 0.2:
        insight_msg = "Moderate Signal: AI sees a move, but indicators are mixed."
    else:
        insight_msg = "Weak Signal: AI and indicators disagree. Market might just move sideways."

    return {
        "direction": lstm_direction,
        "confidence": round(float(final_confidence), 3),
        "insight": insight_msg
    }

def calculate_practical_risk_metrics(historical_prices, predicted_prices):
    """Calculate practical business-oriented risk metrics."""
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
        "risk_reward_ratio": round(float(risk_reward_ratio), 2)
    }

def predict_coin(df, coin):
    """The main orchestrator called by Django tasks."""
    features_list = ["close", "rsi", "macd", "macd_signal", "sma_20", "ema_20", "volume_mean", "atr"]
    
    coin_df = prepare_data(df, coin)
    data_matrix = coin_df[features_list].values
    current_price = float(coin_df["close"].iloc[-1])
    
    model_path, scaler_path = get_model_paths(coin)
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        from tensorflow.keras.models import load_model as keras_load_model
        model = keras_load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        raise Exception(f"AI Model for {coin} is currently training in the background. Please wait.")
        
    preds = lstm_forecast(model, scaler, data_matrix)
    
    try:
        nlp_result = analyze_coin_sentiment(coin)
    except Exception as e:
        nlp_result = {"score": 0.0, "label": "NEUTRAL", "article_count": 0}
        
    # Build indicators dict for genuine confidence scoring
    current_indicators = {
        'rsi': float(coin_df["rsi"].iloc[-1]),
        'macd': float(coin_df["macd"].iloc[-1]),
        'macd_signal': float(coin_df["macd_signal"].iloc[-1]),
        'ema20': float(coin_df["ema_20"].iloc[-1])
    }
    
    decision = ensemble_decision(current_price, preds, nlp_result, current_indicators)
    
    # --- Execute Real Backtest Simulation ---
    backtest_results = run_fast_backtest(model, scaler, data_matrix)
    
    returns = np.diff(preds) / preds[:-1]
    volatility = float(np.std(returns))
    
    recent_historical = coin_df["close"].values[-24:]
    advanced_risk = calculate_practical_risk_metrics(recent_historical, preds)
    
    if advanced_risk["var_95_pct"] > 5.0 or advanced_risk["max_drawdown_pct"] > 10.0:
        risk_level = "HIGH RISK"
    elif advanced_risk["var_95_pct"] > 2.0:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    recent_candles = []
    for _, row in coin_df.tail(100).iterrows():
        recent_candles.append({
            "time": int(row["time"].timestamp()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "value": float(row["volumeto"]) 
        })

    sma20 = float(coin_df["sma_20"].iloc[-1])
    ema20 = float(coin_df["ema_20"].iloc[-1])
        
    return {
        "current_price": current_price,
        "forecast_next_hours": preds,
        "direction": decision["direction"],
        "confidence": decision["confidence"],
        "insight": decision["insight"],
        "volatility": volatility,
        "trend_strength": float(np.mean(returns)),
        "risk_level": risk_level,
        "nlp_score": nlp_result["score"],
        "max_drawdown": advanced_risk["max_drawdown_pct"],
        "value_at_risk": advanced_risk["var_95_pct"],
        "risk_reward": advanced_risk["risk_reward_ratio"],
        "chart_data": recent_candles,
        "mtf_short": decision["direction"],
        "mtf_med": "UP" if current_price > ema20 else "DOWN",
        "mtf_long": "UP" if current_price > sma20 else "DOWN",
        
        # New Backtesting Output Data
        "bt_win_rate": backtest_results["win_rate"],
        "bt_pnl": backtest_results["pnl_pct"],
        "bt_trades": backtest_results["trades"]
    }