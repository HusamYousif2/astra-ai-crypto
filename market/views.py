# from django.http import JsonResponse
# from .services import fetch_all_coins
# from .predict import predict_coin
# from django.shortcuts import render

# def get_market_data(request):
#     """
#     API endpoint to fetch crypto data.
#     """
#     df = fetch_all_coins()

#     # Convert to JSON
#     data = df.to_dict(orient="records")

#     return JsonResponse(data, safe=False)

# def get_ai_analysis(request):
#     df = fetch_all_coins()

#     results = {}

#     for coin in df["symbol"].unique():
#         try:
#             results[coin] = predict_coin(df, coin)
#         except Exception as e:
#             results[coin] = {"error": str(e)}

#     return JsonResponse(results)



# def dashboard(request):
#     return render(request, "market/dashboard.html")
from django.http import JsonResponse
from django.core.cache import cache
from .services import fetch_all_coins
from django.shortcuts import render

def get_market_data(request):
    df = fetch_all_coins()
    data = df.to_dict(orient="records")
    return JsonResponse(data, safe=False)

def get_ai_analysis(request):
    """
    Lightning fast endpoint. 
    It only reads from the cache prepared by the background task.
    """
    cached_results = cache.get('ai_market_analysis')

    if cached_results:
        return JsonResponse(cached_results)

    # Fallback response if the user visits the site EXACTLY when the server 
    # just started and the background task hasn't finished its first run yet.
    return JsonResponse({
        "status": "initializing",
        "message": "AI is analyzing the market in the background. Please refresh in 30 seconds."
    }, status=202)

def dashboard(request):
    return render(request, "market/dashboard.html")