import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..services import fetch_all_coins
from ..models_ai.trainer import (
    build_and_prepare_training_data,
    train_direction_model,
    train_setup_quality_model,
)
from ..models_ai.registry import infer_direction_v2, infer_setup_quality_v2


ONLINE_DIR = Path(__file__).resolve().parent / "online_state"
ONLINE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_FILE = ONLINE_DIR / "rolling_window.pkl"
METRICS_FILE = ONLINE_DIR / "online_metrics.json"
MODEL_STATE_FILE = ONLINE_DIR / "model_state.json"


DEFAULT_MODEL_STATE = {
    "last_full_refresh_at": None,
    "last_fast_update_at": None,
    "last_drift_check_at": None,
    "candidate_direction_model_ready": False,
    "candidate_setup_model_ready": False,
    "production_direction_model": "direction_model_v2",
    "production_setup_model": "setup_quality_model_v2",
}


DEFAULT_METRICS_STATE = {
    "recent_direction_accuracy": None,
    "recent_setup_quality_stability": None,
    "recent_drift_score": None,
    "window_rows": 0,
    "last_updated_at": None,
}


def utc_now_iso():
    """
    Return the current UTC timestamp in ISO format.
    """
    return datetime.now(timezone.utc).isoformat()


def file_exists(path_obj):
    """
    Check whether a file exists.
    """
    return Path(path_obj).exists()


def load_json_file(path_obj, default_value):
    """
    Load a JSON file safely.
    """
    if not file_exists(path_obj):
        return default_value.copy() if isinstance(default_value, dict) else default_value

    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value.copy() if isinstance(default_value, dict) else default_value


def save_json_file(path_obj, payload):
    """
    Save a JSON payload safely.
    """
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_model_state():
    """
    Load model state metadata.
    """
    return load_json_file(MODEL_STATE_FILE, DEFAULT_MODEL_STATE)


def save_model_state(state):
    """
    Save model state metadata.
    """
    save_json_file(MODEL_STATE_FILE, state)


def load_metrics_state():
    """
    Load online metrics state.
    """
    return load_json_file(METRICS_FILE, DEFAULT_METRICS_STATE)


def save_metrics_state(state):
    """
    Save online metrics state.
    """
    save_json_file(METRICS_FILE, state)


def save_rolling_window(df):
    """
    Save the rolling window to disk using pickle.
    """
    df.to_pickle(WINDOW_FILE)


def load_rolling_window():
    """
    Load the rolling window from disk if it exists.
    """
    if not file_exists(WINDOW_FILE):
        return pd.DataFrame()

    try:
        return pd.read_pickle(WINDOW_FILE)
    except Exception:
        return pd.DataFrame()


def deduplicate_market_rows(df):
    """
    Remove duplicate market rows by symbol + time.
    """
    if df.empty:
        return df

    working = df.copy()

    if "symbol" in working.columns and "time" in working.columns:
        working = working.sort_values(["symbol", "time"]).drop_duplicates(
            subset=["symbol", "time"],
            keep="last",
        )

    return working.reset_index(drop=True)


def append_new_market_data(existing_df, new_df):
    """
    Append new market data into the rolling dataset.
    """
    if existing_df is None or existing_df.empty:
        combined = new_df.copy()
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    combined = deduplicate_market_rows(combined)
    return combined


def build_rolling_training_window(df, max_rows_per_coin=1500):
    """
    Keep only the most recent rows per coin.
    """
    if df.empty:
        return df

    if "symbol" not in df.columns:
        return df.tail(max_rows_per_coin).reset_index(drop=True)

    frames = []

    for symbol in df["symbol"].dropna().unique():
        coin_df = df[df["symbol"] == symbol].copy()
        coin_df = coin_df.sort_values("time").tail(max_rows_per_coin)
        frames.append(coin_df)

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    output = deduplicate_market_rows(output)
    return output.reset_index(drop=True)


def estimate_feature_drift(old_df, new_df, columns=None):
    """
    Estimate a simple drift score using mean changes across numeric columns.
    """
    if old_df.empty or new_df.empty:
        return 0.0

    if columns is None:
        numeric_columns = [
            col for col in new_df.columns
            if pd.api.types.is_numeric_dtype(new_df[col])
        ]
    else:
        numeric_columns = columns

    if not numeric_columns:
        return 0.0

    scores = []

    for col in numeric_columns:
        if col not in old_df.columns or col not in new_df.columns:
            continue

        old_series = old_df[col].dropna()
        new_series = new_df[col].dropna()

        if len(old_series) == 0 or len(new_series) == 0:
            continue

        old_mean = float(old_series.mean())
        new_mean = float(new_series.mean())
        baseline = max(abs(old_mean), 1e-8)

        diff_ratio = abs(new_mean - old_mean) / baseline
        scores.append(diff_ratio)

    if not scores:
        return 0.0

    return round(float(np.mean(scores)), 4)


def should_refresh_model(
    metrics_state,
    drift_threshold=0.20,
    min_recent_accuracy=0.48,
):
    """
    Decide whether the model should be refreshed.
    """
    drift_score = metrics_state.get("recent_drift_score")
    recent_accuracy = metrics_state.get("recent_direction_accuracy")

    if drift_score is not None and drift_score >= drift_threshold:
        return True

    if recent_accuracy is not None and recent_accuracy < min_recent_accuracy:
        return True

    return False


def evaluate_recent_direction_proxy(df, sample_symbols=None, max_samples=3):
    """
    Evaluate the currently registered production direction model on recent data.
    This is a lightweight proxy, not a full institutional evaluation.
    """
    if df.empty or "symbol" not in df.columns:
        return None

    symbols = list(df["symbol"].dropna().unique())
    if sample_symbols:
        symbols = [s for s in symbols if s in sample_symbols]

    symbols = symbols[:max_samples]
    if not symbols:
        return None

    results = []

    for symbol in symbols:
        try:
            coin_df = df[df["symbol"] == symbol].copy().sort_values("time")
            if len(coin_df) < 60:
                continue

            inference = infer_direction_v2(coin_df)
            predicted_label = inference.get("label", "NEUTRAL")

            latest_close = float(coin_df.iloc[-1]["close"])
            prev_close = float(coin_df.iloc[-2]["close"])
            realized_return_pct = ((latest_close - prev_close) / prev_close) * 100.0

            if realized_return_pct > 0.20:
                realized_label = "UP"
            elif realized_return_pct < -0.20:
                realized_label = "DOWN"
            else:
                realized_label = "NEUTRAL"

            results.append(1.0 if predicted_label == realized_label else 0.0)

        except Exception:
            continue

    if not results:
        return None

    return round(float(np.mean(results)), 4)


def evaluate_recent_setup_proxy(df, sample_symbols=None, max_samples=3):
    """
    Evaluate the setup-quality model stability on recent data.
    This is a proxy metric, not a final production KPI.
    """
    if df.empty or "symbol" not in df.columns:
        return None

    symbols = list(df["symbol"].dropna().unique())
    if sample_symbols:
        symbols = [s for s in symbols if s in sample_symbols]

    symbols = symbols[:max_samples]
    if not symbols:
        return None

    confidence_scores = []

    for symbol in symbols:
        try:
            coin_df = df[df["symbol"] == symbol].copy().sort_values("time")
            if len(coin_df) < 60:
                continue

            inference = infer_setup_quality_v2(coin_df)
            score = inference.get("score", 0.0)
            confidence_scores.append(score / 100.0)

        except Exception:
            continue

    if not confidence_scores:
        return None

    return round(float(np.mean(confidence_scores)), 4)


def refresh_online_metrics(previous_window, latest_window):
    """
    Refresh drift and proxy metrics after the rolling window update.
    """
    metrics_state = load_metrics_state()

    drift_score = estimate_feature_drift(previous_window, latest_window)

    recent_direction_accuracy = evaluate_recent_direction_proxy(latest_window)
    recent_setup_quality_stability = evaluate_recent_setup_proxy(latest_window)

    metrics_state["recent_drift_score"] = drift_score
    metrics_state["recent_direction_accuracy"] = recent_direction_accuracy
    metrics_state["recent_setup_quality_stability"] = recent_setup_quality_stability
    metrics_state["window_rows"] = int(len(latest_window))
    metrics_state["last_updated_at"] = utc_now_iso()

    save_metrics_state(metrics_state)
    return metrics_state


def train_candidate_models(fetch_callable):
    """
    Train candidate V2 models on the latest rolling dataset.
    """
    training_df = build_and_prepare_training_data(fetch_callable)

    direction_result = train_direction_model(training_df)
    setup_result = train_setup_quality_model(training_df)

    model_state = load_model_state()
    model_state["candidate_direction_model_ready"] = True
    model_state["candidate_setup_model_ready"] = True
    model_state["last_full_refresh_at"] = utc_now_iso()
    save_model_state(model_state)

    return {
        "direction_result": direction_result,
        "setup_result": setup_result,
    }


def run_fast_online_update(fetch_callable=fetch_all_coins, max_rows_per_coin=1500):
    """
    Run a fast online update:
    - fetch latest market data
    - merge with rolling window
    - recompute drift and proxy metrics
    - decide whether refresh is needed
    """
    previous_window = load_rolling_window()
    latest_data = fetch_callable()

    combined = append_new_market_data(previous_window, latest_data)
    rolling_window = build_rolling_training_window(
        combined,
        max_rows_per_coin=max_rows_per_coin,
    )
    save_rolling_window(rolling_window)

    metrics_state = refresh_online_metrics(previous_window, rolling_window)

    model_state = load_model_state()
    model_state["last_fast_update_at"] = utc_now_iso()
    model_state["last_drift_check_at"] = utc_now_iso()
    save_model_state(model_state)

    refresh_needed = should_refresh_model(metrics_state)

    return {
        "rolling_window_rows": int(len(rolling_window)),
        "metrics_state": metrics_state,
        "refresh_needed": refresh_needed,
    }


def promote_candidate_model_if_better():
    """
    Promote candidate models logically.
    In this baseline implementation, we mark them as production-ready once trained.
    Later this can be extended with challenger-vs-champion evaluation.
    """
    model_state = load_model_state()

    if model_state.get("candidate_direction_model_ready"):
        model_state["production_direction_model"] = "direction_model_v2"

    if model_state.get("candidate_setup_model_ready"):
        model_state["production_setup_model"] = "setup_quality_model_v2"

    save_model_state(model_state)
    return model_state


def run_incremental_update(
    fetch_callable=fetch_all_coins,
    retrain_if_needed=True,
):
    """
    Full online adaptation step:
    1. Update rolling window
    2. Recompute drift and proxy metrics
    3. Retrain if needed
    4. Promote candidate models if appropriate
    """
    online_result = run_fast_online_update(fetch_callable=fetch_callable)

    output = {
        "online_result": online_result,
        "retrained": False,
        "promotion_state": None,
    }

    if retrain_if_needed and online_result["refresh_needed"]:
        train_candidate_models(fetch_callable)
        output["retrained"] = True
        output["promotion_state"] = promote_candidate_model_if_better()

    return output


def initialize_online_learning(fetch_callable=fetch_all_coins):
    """
    Initialize online learning storage from scratch.
    """
    latest_data = fetch_callable()
    rolling_window = build_rolling_training_window(latest_data)
    save_rolling_window(rolling_window)

    save_metrics_state({
        **DEFAULT_METRICS_STATE,
        "window_rows": int(len(rolling_window)),
        "last_updated_at": utc_now_iso(),
    })

    save_model_state({
        **DEFAULT_MODEL_STATE,
        "last_fast_update_at": utc_now_iso(),
        "last_drift_check_at": utc_now_iso(),
    })

    return {
        "initialized": True,
        "window_rows": int(len(rolling_window)),
    }


if __name__ == "__main__":
    print("online_learning.py loaded successfully.")