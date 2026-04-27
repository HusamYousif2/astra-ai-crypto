import logging
import os

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
from .predict_v2 import predict_coin_v2
from .services import fetch_all_coins, save_market_candles

logging.getLogger("absl").setLevel(logging.ERROR)

scheduler = None
startup_warmup_done = False


def safe_float_for_db(value, default=0.0):
    if value is None:
        return default

    try:
        return float(value)
    except Exception:
        return default


def background_ai_training():
    print("\n[SYSTEM] Starting scheduled AI background training & cache warm-up...")

    try:
        df = fetch_all_coins(interval="1h", months=6)
        save_market_candles(df, interval="1h")

        coins = df["symbol"].unique()

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
                print(f"--> Fine-tuning {coin} legacy forecast model with latest data...")
                model = keras_load_model(model_path)
                scaler = joblib.load(scaler_path)

                recent_data = data_matrix[-(SEQ_LENGTH + 48):]
                if len(recent_data) > SEQ_LENGTH:
                    scaled_data = scaler.transform(recent_data)
                    X_new, y_new = create_sequences(scaled_data, SEQ_LENGTH)

                    if len(X_new) > 0:
                        model.compile(optimizer="adam", loss="mean_squared_error")
                        model.fit(X_new, y_new, epochs=1, batch_size=4, verbose=0)
                        model.save(model_path)
            else:
                print(f"--> No legacy forecast model found for {coin}. Training from scratch...")
                train_lstm_for_coin(df, coin)

        print("[SYSTEM] Models updated. Generating new V2 predictions for cache...")
        results = {}

        for coin in coins:
            try:
                result = predict_coin_v2(df, coin)
                results[coin] = result

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
                    current_price=safe_float_for_db(result.get("current_price")),
                    direction=result.get("direction", "NEUTRAL"),
                    confidence=safe_float_for_db(result.get("confidence")),
                    insight=result.get("insight", ""),
                    risk_level=result.get("risk_level", "MEDIUM RISK"),
                    nlp_score=safe_float_for_db(result.get("nlp_score")),
                    max_drawdown=safe_float_for_db(result.get("max_drawdown")),
                    value_at_risk=safe_float_for_db(result.get("value_at_risk")),
                    risk_reward=safe_float_for_db(result.get("risk_reward")),
                )

            except Exception as e:
                print(f"Error predicting {coin} with V2: {e}")
                results[coin] = {"error": str(e)}

        cache.set("ai_market_analysis", results, 1200)
        print("[SYSTEM] V2 cache successfully updated! Dashboard is now instantly ready.\n")

    except Exception as e:
        print(f"[ERROR] Background task failed: {e}")


def should_enable_scheduler():
    if os.environ.get("DISABLE_SCHEDULER", "False") == "True":
        return False

    if os.environ.get("RENDER") == "true":
        return False

    return True


def start_scheduler():
    global scheduler
    global startup_warmup_done

    if not should_enable_scheduler():
        print("[SYSTEM] Scheduler disabled for this environment.")
        return

    if scheduler is not None:
        print("[SYSTEM] Scheduler already running.")
        return

    if os.environ.get("RUN_MAIN"):
        print("[SYSTEM] Starting scheduler...")

        if not startup_warmup_done:
            print("[SYSTEM] Running one immediate startup warm-up...")
            background_ai_training()
            startup_warmup_done = True

        scheduler = BackgroundScheduler()

        scheduler.add_job(
            background_ai_training,
            "interval",
            minutes=15,
            id="ai_training_job",
            replace_existing=True,
        )

        scheduler.start()
        print("[SYSTEM] Scheduler started successfully.")