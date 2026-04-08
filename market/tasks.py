# market/tasks.py

import os
import logging
from datetime import datetime

import joblib
from apscheduler.schedulers.background import BackgroundScheduler  # type: ignore
from django.core.cache import cache
from tensorflow.keras.models import load_model as keras_load_model  # type: ignore

from .features import add_indicators
from .lstm_model import (
    SEQ_LENGTH,
    create_sequences,
    get_model_paths,
    train_lstm_for_coin,
)
from .models import MarketPrediction
from .predict import predict_coin
from .services import fetch_all_coins

# Suppress warnings in background tasks
logging.getLogger("absl").setLevel(logging.ERROR)


def background_ai_training():
    """
    Scheduled task to fetch fresh data, fine-tune models,
    generate new predictions, update the cache,
    and store prediction snapshots in the database.
    """
    print("\n[SYSTEM] Starting scheduled AI background training & cache warm-up...")

    try:
        df = fetch_all_coins()
        coins = df["symbol"].unique()

        # Phase 1: Online learning and fine-tuning
        for coin in coins:
            model_path, scaler_path = get_model_paths(coin)

            coin_df = df[df["symbol"] == coin].copy()
            coin_df = add_indicators(coin_df)
            coin_df = coin_df.dropna()

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
            data_matrix = coin_df[features_list].values

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print(f"--> Fine-tuning {coin} model with latest data...")
                model = keras_load_model(model_path)
                scaler = joblib.load(scaler_path)

                recent_data = data_matrix[-48:]
                if len(recent_data) > SEQ_LENGTH:
                    scaled_data = scaler.transform(recent_data)
                    X_new, y_new = create_sequences(scaled_data, SEQ_LENGTH)

                    if len(X_new) > 0:
                        model.compile(optimizer="adam", loss="mean_squared_error")
                        model.fit(X_new, y_new, epochs=1, batch_size=4, verbose=0)
                        model.save(model_path)
            else:
                print(f"--> No model found for {coin}. Training from scratch...")
                train_lstm_for_coin(df, coin)

        # Phase 2: Cache warming and database snapshot storage
        print("[SYSTEM] Models updated. Generating new predictions for cache...")
        results = {}

        for coin in coins:
            try:
                result = predict_coin(df, coin)
                results[coin] = result

                # Remove the latest duplicate snapshot for the same symbol if needed
                last_entry = (
                    MarketPrediction.objects.filter(symbol=coin)
                    .order_by("-created_at")
                    .first()
                )

                if last_entry:
                    same_direction = last_entry.direction == result["direction"]
                    same_price = abs(last_entry.current_price - result["current_price"]) < 0.0001
                    same_confidence = abs(last_entry.confidence - result["confidence"]) < 0.0001

                    if same_direction and same_price and same_confidence:
                        last_entry.delete()

                MarketPrediction.objects.create(
                    symbol=coin,
                    current_price=result["current_price"],
                    direction=result["direction"],
                    confidence=result["confidence"],
                    insight=result["insight"],
                    risk_level=result["risk_level"],
                    nlp_score=result.get("nlp_score", 0.0),
                    max_drawdown=result.get("max_drawdown", 0.0),
                    value_at_risk=result.get("value_at_risk", 0.0),
                    risk_reward=result.get("risk_reward", 0.0),
                )

            except Exception as e:
                print(f"Error predicting {coin}: {e}")
                results[coin] = {"error": str(e)}

        # Save to cache for 20 minutes
        cache.set("ai_market_analysis", results, 1200)

        print("[SYSTEM] Cache successfully updated! Dashboard is now instantly ready.\n")

    except Exception as e:
        print(f"[ERROR] Background task failed: {e}")


def start_scheduler():
    if os.environ.get("RUN_MAIN"):
        scheduler = BackgroundScheduler()

        scheduler.add_job(
            background_ai_training,
            "interval",
            minutes=15,
            id="ai_training_job",
            replace_existing=True,
            next_run_time=datetime.now(),
        )
        scheduler.start()
        print("[SYSTEM] APScheduler started. First run executing now...")