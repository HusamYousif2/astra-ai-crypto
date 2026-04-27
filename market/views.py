from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render

from .models import MarketPrediction
from .services import fetch_all_coins, get_market_history


def get_market_data(request):
    interval = request.GET.get("interval", "1h")
    months = int(request.GET.get("months", "6"))

    df = fetch_all_coins(interval=interval, months=months)
    data = df.to_dict(orient="records")
    return JsonResponse(data, safe=False)


def get_market_history_api(request):
    symbol = request.GET.get("symbol", "BTC").upper()
    interval = request.GET.get("interval", "1h")
    range_key = request.GET.get("range", "6m")

    if interval not in {"1h", "4h", "1d"}:
        return JsonResponse({"error": "Unsupported interval"}, status=400)

    if range_key not in {"1d", "1w", "1m", "3m", "6m"}:
        return JsonResponse({"error": "Unsupported range"}, status=400)

    data = get_market_history(
        symbol=symbol,
        interval=interval,
        range_key=range_key,
    )

    return JsonResponse(
        {
            "symbol": symbol,
            "interval": interval,
            "range": range_key,
            "candles": data,
        }
    )


def build_latest_predictions_from_db():
    """
    Fallback payload builder when cache is empty.
    It reconstructs a lightweight analysis object from the latest DB snapshots.
    """
    symbols = (
        MarketPrediction.objects.order_by()
        .values_list("symbol", flat=True)
        .distinct()
    )

    results = {}

    for symbol in symbols:
        item = (
            MarketPrediction.objects.filter(symbol=symbol)
            .order_by("-created_at")
            .first()
        )

        if not item:
            continue

        results[symbol] = {
            "current_price": item.current_price,
            "forecast_next_hours": [],
            "direction": item.direction,
            "confidence": item.confidence,
            "insight": item.insight,
            "volatility": 0.0,
            "trend_strength": 0.0,
            "risk_level": item.risk_level,
            "nlp_score": item.nlp_score,
            "max_drawdown": item.max_drawdown,
            "value_at_risk": item.value_at_risk,
            "risk_reward": item.risk_reward,
            "chart_data": [],
            "mtf_short": "MIXED",
            "mtf_med": "MIXED",
            "mtf_long": "MIXED",
            "bt_win_rate": 0,
            "bt_pnl": 0,
            "bt_trades": 0,
            "signal": "HOLD" if item.direction == "NEUTRAL" else ("BUY" if item.direction == "UP" else "SELL"),
            "signal_strength": "Weak",
            "trade_stance": "No Trade",
            "setup_score": round(float(item.confidence * 100), 1),
            "market_regime": "Transitional",
            "risk_posture": item.risk_level.replace(" RISK", "").replace(" risk", "").upper(),
            "confidence_explainer": "This score reflects the strength and clarity of the current trading setup. It does not describe the accuracy of the current market price.",
            "signal_explainer": item.insight,
            "ai_summary": item.insight,
            "bullish_factors": [],
            "bearish_factors": [],
            "watchpoints": [],
            "scenario_analysis": {
                "bull_case": 0.0,
                "base_case": 0.0,
                "bear_case": 0.0,
            },
            "news_context": {
                "sentiment_score": item.nlp_score,
                "sentiment_label": "NEUTRAL",
                "article_count": 0,
                "summary": "Cached live analysis is warming up. Showing the latest saved database snapshot.",
                "top_articles": [],
            },
            "technical_context": {
                "rsi": 0.0,
                "macd": 0.0,
                "macd_signal": 0.0,
                "ema20": 0.0,
                "sma20": 0.0,
                "atr": 0.0,
            },
            "created_at": item.created_at.isoformat(),
        }

    return results


def get_ai_analysis(request):
    """
    Fast endpoint:
    1) Try cache
    2) Fallback to latest DB snapshots
    3) Otherwise return initializing
    """
    cached_results = cache.get("ai_market_analysis")

    if cached_results:
        return JsonResponse(cached_results)

    db_results = build_latest_predictions_from_db()
    if db_results:
        return JsonResponse(db_results)

    return JsonResponse(
        {
            "status": "initializing",
            "message": "AI is analyzing the market in the background. Please refresh in 30 seconds.",
        },
        status=202,
    )


def dashboard(request):
    return render(request, "market/dashboard.html")


def get_prediction_history(request):
    latest_predictions = MarketPrediction.objects.all()[:20]

    data = []
    for item in latest_predictions:
        data.append(
            {
                "symbol": item.symbol,
                "current_price": item.current_price,
                "direction": item.direction,
                "confidence": item.confidence,
                "insight": item.insight,
                "risk_level": item.risk_level,
                "nlp_score": item.nlp_score,
                "max_drawdown": item.max_drawdown,
                "value_at_risk": item.value_at_risk,
                "risk_reward": item.risk_reward,
                "created_at": item.created_at,
            }
        )

    return JsonResponse(data, safe=False)


def get_latest_predictions(request):
    symbols = (
        MarketPrediction.objects.order_by()
        .values_list("symbol", flat=True)
        .distinct()
    )
    data = []

    for symbol in symbols:
        item = (
            MarketPrediction.objects.filter(symbol=symbol)
            .order_by("-created_at")
            .first()
        )

        if item:
            data.append(
                {
                    "symbol": item.symbol,
                    "current_price": item.current_price,
                    "direction": item.direction,
                    "confidence": item.confidence,
                    "insight": item.insight,
                    "risk_level": item.risk_level,
                    "nlp_score": item.nlp_score,
                    "max_drawdown": item.max_drawdown,
                    "value_at_risk": item.value_at_risk,
                    "risk_reward": item.risk_reward,
                    "created_at": item.created_at,
                }
            )

    return JsonResponse(data, safe=False)