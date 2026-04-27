from django.contrib import admin
from django.urls import path
from market.views import (
    dashboard,
    get_ai_analysis,
    get_latest_predictions,
    get_market_data,
    get_market_history_api,
    get_prediction_history,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/market-data/", get_market_data),
    path("api/market-history/", get_market_history_api),
    path("api/ai-analysis/", get_ai_analysis),
    path("api/prediction-history/", get_prediction_history),
    path("api/latest-predictions/", get_latest_predictions),
    path("", dashboard),
]