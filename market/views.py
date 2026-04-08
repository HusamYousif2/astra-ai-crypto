from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render

from .models import MarketPrediction
from .services import fetch_all_coins


def get_market_data(request):
    df = fetch_all_coins()
    data = df.to_dict(orient="records")
    return JsonResponse(data, safe=False)


def get_ai_analysis(request):
    """
    Lightning fast endpoint.
    It only reads from the cache prepared by the background task.
    """
    cached_results = cache.get("ai_market_analysis")

    if cached_results:
        return JsonResponse(cached_results)

    # Return a fallback response while background analysis is still running
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