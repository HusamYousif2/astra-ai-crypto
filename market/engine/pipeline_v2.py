import pandas as pd

from .decision_engine import build_decision_snapshot
from .features_v2 import build_features_v2, build_features_v2_for_all_coins
from .labels import build_training_labels_for_all_coins
from .news_engine_v2 import build_news_context_v2
from .risk_engine import add_risk_columns, get_latest_risk_snapshot
from .state_engine import add_market_state_columns, get_latest_state_snapshot


TECHNICAL_CONTEXT_COLUMNS_V2 = [
    "rsi",
    "macd",
    "macd_signal",
    "ema_20",
    "sma_20",
    "atr",
    "atr_pct",
    "ret_1",
    "ret_6",
    "ret_24",
    "realized_vol_24",
    "volume_vs_ma24",
    "range_pct",
    "trend_alignment_flag",
    "compression_flag",
    "expansion_flag",
]


def ensure_dataframe_not_empty(df, message):
    """
    Validate that a DataFrame is not empty.
    """
    if df is None or df.empty:
        raise ValueError(message)


def filter_coin_dataframe(df, coin):
    """
    Filter a multi-coin DataFrame to one symbol.
    """
    if "symbol" not in df.columns:
        return df.copy()

    coin_df = df[df["symbol"] == coin].copy()
    return coin_df


def infer_forecast_context(current_price, forecast_values):
    """
    Build a lightweight forecast context from a list of future prices.
    """
    if forecast_values is None or len(forecast_values) == 0:
        return {}

    final_price = forecast_values[-1]
    expected_move_pct = ((final_price - current_price) / current_price) * 100.0

    if expected_move_pct > 0:
        direction = "UP"
    elif expected_move_pct < 0:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"

    return {
        "expected_move_pct": round(float(expected_move_pct), 3),
        "direction": direction,
        "forecast_last": float(final_price),
        "forecast_max": float(max(forecast_values)),
        "forecast_min": float(min(forecast_values)),
    }


def build_chart_payload(df, limit=100):
    """
    Convert the latest candles into a chart-friendly payload.
    """
    rows = []

    for _, row in df.tail(limit).iterrows():
        rows.append(
            {
                "time": int(row["time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "value": float(row["volumeto"]) if "volumeto" in row else 0.0,
            }
        )

    return rows


def extract_technical_context_v2(row):
    """
    Extract a clean technical context dictionary for UI or API use.
    """
    context = {}

    for column in TECHNICAL_CONTEXT_COLUMNS_V2:
        value = row.get(column, None)

        if value is None or pd.isna(value):
            context[column] = None
        else:
            context[column] = round(float(value), 6)

    return context


def build_live_snapshot_v2_for_coin(
    df,
    coin,
    news_context=None,
    forecast_values=None,
):
    """
    Build a live V2 intelligence snapshot for a single coin.
    """
    coin_df = filter_coin_dataframe(df, coin)
    ensure_dataframe_not_empty(
        coin_df,
        f"No data available for coin: {coin}",
    )

    featured_df = build_features_v2(coin_df)
    ensure_dataframe_not_empty(
        featured_df,
        f"Feature engineering produced no rows for coin: {coin}",
    )

    stated_df = add_market_state_columns(featured_df)
    risk_df = add_risk_columns(stated_df)
    ensure_dataframe_not_empty(
        risk_df,
        f"Risk pipeline produced no rows for coin: {coin}",
    )

    latest_row = risk_df.iloc[-1].to_dict()
    current_price = float(latest_row["close"])

    forecast_context = infer_forecast_context(
        current_price=current_price,
        forecast_values=forecast_values,
    )

    market_state_label = latest_row.get("market_state_label", "Transitional")

    # Build news context automatically if not provided
    if news_context is None:
        news_context = build_news_context_v2(
            coin_symbol=coin,
            market_state_label=market_state_label,
        )

    decision_snapshot = build_decision_snapshot(
        latest_row,
        news_context=news_context,
        forecast_context=forecast_context,
    )

    state_snapshot = get_latest_state_snapshot(risk_df)
    risk_snapshot = get_latest_risk_snapshot(risk_df)

    snapshot = {
        "symbol": coin,
        "current_price": current_price,
        "timestamp": latest_row["time"].isoformat(),
        "market_state": state_snapshot,
        "risk_snapshot": risk_snapshot,
        "decision_snapshot": decision_snapshot,
        "technical_context_v2": extract_technical_context_v2(latest_row),
        "chart_data": build_chart_payload(risk_df),
        "forecast_context": forecast_context,
        "news_context": news_context or {},
    }

    return snapshot


def build_live_snapshots_v2_for_all_coins(
    df,
    news_context_map=None,
    forecast_map=None,
):
    """
    Build V2 live intelligence snapshots for all coins in the input DataFrame.
    """
    ensure_dataframe_not_empty(
        df,
        "Input DataFrame is empty. Cannot build live snapshots.",
    )

    if "symbol" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'symbol' column.")

    results = {}
    news_context_map = news_context_map or {}
    forecast_map = forecast_map or {}

    for coin in df["symbol"].dropna().unique():
        try:
            provided_news_context = news_context_map.get(coin)

            results[coin] = build_live_snapshot_v2_for_coin(
                df=df,
                coin=coin,
                news_context=provided_news_context,
                forecast_values=forecast_map.get(coin),
            )
        except Exception as e:
            results[coin] = {"error": str(e)}

    return results


def prepare_training_dataset_v2(
    df,
    horizon=6,
    neutral_threshold=0.20,
):
    """
    Build a full V2 training dataset:
    1. Generate features V2
    2. Generate labels for supervised learning
    """
    ensure_dataframe_not_empty(
        df,
        "Input DataFrame is empty. Cannot prepare training dataset.",
    )

    featured_df = build_features_v2_for_all_coins(df)
    ensure_dataframe_not_empty(
        featured_df,
        "Feature engineering V2 produced no rows.",
    )

    labeled_df = build_training_labels_for_all_coins(
        featured_df,
        horizon=horizon,
        neutral_threshold=neutral_threshold,
    )
    ensure_dataframe_not_empty(
        labeled_df,
        "Label generation produced no rows.",
    )

    return labeled_df


def summarize_pipeline_v2_snapshot(snapshot):
    """
    Build a lighter summary from a full V2 snapshot.
    """
    if not snapshot:
        return {}

    decision = snapshot.get("decision_snapshot", {})
    market_state = snapshot.get("market_state", {})
    risk_snapshot = snapshot.get("risk_snapshot", {})
    news_context = snapshot.get("news_context", {})

    return {
        "symbol": snapshot.get("symbol"),
        "current_price": snapshot.get("current_price"),
        "timestamp": snapshot.get("timestamp"),
        "signal": decision.get("signal"),
        "signal_strength": decision.get("signal_strength"),
        "trade_stance": decision.get("trade_stance"),
        "setup_score": decision.get("setup_score"),
        "market_regime": market_state.get("market_state_label"),
        "risk_level": risk_snapshot.get("risk_level"),
        "trade_safety": risk_snapshot.get("trade_safety"),
        "news_impact": decision.get("news_impact"),
        "news_label": news_context.get("sentiment_label"),
    }


if __name__ == "__main__":
    print("pipeline_v2.py loaded successfully.")