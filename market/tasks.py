# market/tasks.py

import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from tensorflow.keras.models import load_model as keras_load_model # type: ignore
import joblib
from django.core.cache import cache

from datetime import datetime

from .services import fetch_all_coins
from .features import add_indicators
from .lstm_model import get_model_paths, train_lstm_for_coin, create_sequences, SEQ_LENGTH
from .predict import predict_coin

# Suppress warnings in background tasks
logging.getLogger('absl').setLevel(logging.ERROR)

def background_ai_training():
    """
    Scheduled task to fetch fresh data, fine-tune models, 
    generate new predictions, and update the fast cache.
    """
    print("\n[SYSTEM] Starting scheduled AI background training & Cache Warm-up...")
    
    try:
        df = fetch_all_coins()
        coins = df["symbol"].unique()
        
        # --- PHASE 1: Online Learning & Fine-Tuning ---
        for coin in coins:
            model_path, scaler_path = get_model_paths(coin)
            
            coin_df = df[df["symbol"] == coin].copy()
            coin_df = add_indicators(coin_df)
            coin_df = coin_df.dropna()
            
            features_list = ["close", "rsi", "macd", "macd_signal", "sma_20", "ema_20", "volume_mean", "atr"]
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
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        model.fit(X_new, y_new, epochs=1, batch_size=4, verbose=0)
                        model.save(model_path)
            else:
                print(f"--> No model found for {coin}. Training from scratch...")
                train_lstm_for_coin(df, coin)
                
        # --- PHASE 2: Cache Warming (Generating final predictions) ---
        print("[SYSTEM] Models updated. Generating new predictions for Cache...")
        results = {}
        
        for coin in coins:
            try:
                results[coin] = predict_coin(df, coin)
            except Exception as e:
                print(f"Error predicting {coin}: {e}")
                results[coin] = {"error": str(e)}

        # Save to cache for 20 minutes (1200 seconds)
        # This overlaps safely until the next 15-minute job finishes
        cache.set('ai_market_analysis', results, 1200)
        
        print("[SYSTEM] Cache successfully updated! Dashboard is now instantly ready.\n")
        
    except Exception as e:
        print(f"[ERROR] Background task failed: {e}")

def start_scheduler():
    if os.environ.get('RUN_MAIN'):
        scheduler = BackgroundScheduler()
        
        scheduler.add_job(
            background_ai_training, 
            'interval', 
            minutes=15, 
            id='ai_training_job', 
            replace_existing=True,
            next_run_time=datetime.now() 
        )
        scheduler.start()
        print("[SYSTEM] APScheduler started. First run executing NOW...")